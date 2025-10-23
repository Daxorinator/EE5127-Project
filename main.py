import asyncio
import struct
import io
import logging
from bleak import BleakScanner, BleakClient
from collections import deque
import statistics
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread, Lock
import numpy as np

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

# Calibration settings
CALIBRATION_SAMPLES = 50  # Number of samples to average for calibration
NOISE_THRESHOLD = 0.1     # Values below this are considered noise (m/s^2 or dps)

# Windowing settings
WINDOW_SIZE = 128         # 2.56 seconds at 50 Hz
WINDOW_STEP = 64          # 50% overlap (step by half the window size)
SAMPLE_RATE = 50          # Hz

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a custom formatter with no prefix
csv_formatter = logging.Formatter('%(message)s')

# Create file handler that dumps logged data to CSV
file_handler = logging.FileHandler('data.csv', mode='w')
file_handler.setFormatter(csv_formatter)
logger.addHandler(file_handler)

# Create console handler that outputs normal logs to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class PlotData:
    """Thread-safe container for plot data"""
    def __init__(self):
        self.lock = Lock()
        
        # Current window data (only the latest window)
        self.current_window = None
        self.window_timestamps = None
        
        # Historical window means for trending
        self.window_means_ax = deque(maxlen=50)  # Keep last 50 windows
        self.window_means_ay = deque(maxlen=50)
        self.window_means_az = deque(maxlen=50)
        self.window_means_gx = deque(maxlen=50)
        self.window_means_gy = deque(maxlen=50)
        self.window_means_gz = deque(maxlen=50)
        self.window_ids = deque(maxlen=50)
        
        self.window_count = 0
    
    def update_window(self, window_data):
        """Update with a new complete window"""
        with self.lock:
            # Convert window to numpy array
            window_array = np.array(window_data)
            self.current_window = window_array
            self.window_timestamps = np.arange(len(window_data)) / SAMPLE_RATE
            
            # Calculate means for this window
            means = np.mean(window_array, axis=0)
            self.window_means_ax.append(means[0])
            self.window_means_ay.append(means[1])
            self.window_means_az.append(means[2])
            self.window_means_gx.append(means[3])
            self.window_means_gy.append(means[4])
            self.window_means_gz.append(means[5])
            
            self.window_ids.append(self.window_count)
            self.window_count += 1
    
    def get_current_window(self):
        """Get current window data (thread-safe)"""
        with self.lock:
            if self.current_window is None:
                return None, None
            return self.current_window.copy(), self.window_timestamps.copy()
    
    def get_window_means(self):
        """Get historical window means (thread-safe)"""
        with self.lock:
            if len(self.window_ids) == 0:
                return None
            return {
                'ids': list(self.window_ids),
                'ax': list(self.window_means_ax),
                'ay': list(self.window_means_ay),
                'az': list(self.window_means_az),
                'gx': list(self.window_means_gx),
                'gy': list(self.window_means_gy),
                'gz': list(self.window_means_gz)
            }

# Global plot data
plot_data = PlotData()

class SensorProcessor:
    def __init__(self):
        # Calibration offsets
        self.ax_offset = 0.0
        self.ay_offset = 0.0
        self.az_offset = 0.0
        self.gx_offset = 0.0
        self.gy_offset = 0.0
        self.gz_offset = 0.0
        
        # Calibration data collection
        self.calibration_data = []
        self.is_calibrated = False
        
        # Sliding window buffer
        self.window_buffer = deque(maxlen=WINDOW_SIZE)
        self.samples_since_last_window = 0
        
    def add_calibration_sample(self, ax, ay, az, gx, gy, gz):
        """Collect samples for calibration"""
        self.calibration_data.append((ax, ay, az, gx, gy, gz))
        
        if len(self.calibration_data) >= CALIBRATION_SAMPLES:
            self._compute_offsets()
            return True
        return False
    
    def _compute_offsets(self):
        """Compute average offsets from calibration samples"""
        ax_samples = [s[0] for s in self.calibration_data]
        ay_samples = [s[1] for s in self.calibration_data]
        az_samples = [s[2] for s in self.calibration_data]
        gx_samples = [s[3] for s in self.calibration_data]
        gy_samples = [s[4] for s in self.calibration_data]
        gz_samples = [s[5] for s in self.calibration_data]
        
        self.ax_offset = statistics.mean(ax_samples)
        self.ay_offset = statistics.mean(ay_samples)
        self.az_offset = statistics.mean(az_samples)
        self.gx_offset = statistics.mean(gx_samples)
        self.gy_offset = statistics.mean(gy_samples)
        self.gz_offset = statistics.mean(gz_samples)
        
        self.is_calibrated = True
        logger.info(f"Calibration complete: ax={self.ax_offset:.3f}, ay={self.ay_offset:.3f}, "
                   f"az={self.az_offset:.3f}, gx={self.gx_offset:.3f}, gy={self.gy_offset:.3f}, "
                   f"gz={self.gz_offset:.3f}")
    
    def apply_noise_filter(self, value):
        """Apply dead-zone filter to remove small noise"""
        return 0.0 if abs(value) < NOISE_THRESHOLD else value
    
    def process(self, ax, ay, az, gx, gy, gz):
        """Apply calibration and noise filtering"""
        # Apply offsets
        ax_cal = ax - self.ax_offset
        ay_cal = ay - self.ay_offset
        az_cal = az - self.az_offset
        gx_cal = gx - self.gx_offset
        gy_cal = gy - self.gy_offset
        gz_cal = gz - self.gz_offset
        
        # Apply noise filter
        ax_filtered = self.apply_noise_filter(ax_cal)
        ay_filtered = self.apply_noise_filter(ay_cal)
        az_filtered = self.apply_noise_filter(az_cal)
        gx_filtered = self.apply_noise_filter(gx_cal)
        gy_filtered = self.apply_noise_filter(gy_cal)
        gz_filtered = self.apply_noise_filter(gz_cal)
        
        return ax_filtered, ay_filtered, az_filtered, gx_filtered, gy_filtered, gz_filtered
    
    def add_to_window(self, ax, ay, az, gx, gy, gz):
        """Add sample to sliding window and return window when ready"""
        self.window_buffer.append((ax, ay, az, gx, gy, gz))
        self.samples_since_last_window += 1
        
        # Check if we have a full window and have stepped far enough
        if len(self.window_buffer) == WINDOW_SIZE and self.samples_since_last_window >= WINDOW_STEP:
            self.samples_since_last_window = 0
            # Return a copy of the current window
            return list(self.window_buffer)
        
        return None

# Global sensor processor
sensor_processor = SensorProcessor()

def handle_notification(_, data: bytearray):
    try:
        ax, ay, az, gx, gy, gz = struct.unpack("<ffffff", data)
        
        # Calibration phase
        if not sensor_processor.is_calibrated:
            calibrated = sensor_processor.add_calibration_sample(ax, ay, az, gx, gy, gz)
            if calibrated:
                logger.info("window_id,sample_id,acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z")
            return
        
        # Process data with calibration and filtering
        ax_f, ay_f, az_f, gx_f, gy_f, gz_f = sensor_processor.process(ax, ay, az, gx, gy, gz)
        
        # Add to sliding window
        window = sensor_processor.add_to_window(ax_f, ay_f, az_f, gx_f, gy_f, gz_f)
        
        # If we have a complete window, log it and update plot
        if window is not None:
            window_id = (len(sensor_processor.window_buffer) - WINDOW_SIZE) // WINDOW_STEP + 1
            for sample_id, (ax, ay, az, gx, gy, gz) in enumerate(window):
                logger.info(f"{window_id},{sample_id},{ax:.1f},{ay:.1f},{az:.1f},{gx:.1f},{gy:.1f},{gz:.1f}")
            
            # Update plot data with new window
            plot_data.update_window(window)
    except Exception as e:
        logger.error(f"Decode error: {e}")

def setup_plot_window1():
    """Setup first plotting window - current window raw data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Window 1: Current Window Data')
    
    # Accelerometer plot
    line_ax, = ax1.plot([], [], 'r-', label='X', linewidth=1.5)
    line_ay, = ax1.plot([], [], 'g-', label='Y', linewidth=1.5)
    line_az, = ax1.plot([], [], 'b-', label='Z', linewidth=1.5)
    ax1.set_xlim(0, WINDOW_SIZE / SAMPLE_RATE)
    ax1.set_ylim(-20, 20)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Current Window - Accelerometer Data')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Gyroscope plot
    line_gx, = ax2.plot([], [], 'r-', label='X', linewidth=1.5)
    line_gy, = ax2.plot([], [], 'g-', label='Y', linewidth=1.5)
    line_gz, = ax2.plot([], [], 'b-', label='Z', linewidth=1.5)
    ax2.set_xlim(0, WINDOW_SIZE / SAMPLE_RATE)
    ax2.set_ylim(-40, 40)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Current Window - Gyroscope Data')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    lines = {
        'ax': line_ax, 'ay': line_ay, 'az': line_az,
        'gx': line_gx, 'gy': line_gy, 'gz': line_gz
    }
    
    return fig, ax1, ax2, lines

def setup_plot_window2():
    """Setup second plotting window - means and FFT"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Window 2: Trends and FFT Analysis')
    
    # Window means plot
    line_ax, = ax1.plot([], [], 'r-', label='Accel X', linewidth=1.5, marker='o', markersize=3)
    line_ay, = ax1.plot([], [], 'g-', label='Accel Y', linewidth=1.5, marker='o', markersize=3)
    line_az, = ax1.plot([], [], 'b-', label='Accel Z', linewidth=1.5, marker='o', markersize=3)
    ax1.set_xlabel('Window ID')
    ax1.set_ylabel('Mean Acceleration (m/s²)')
    ax1.set_ylim(-20, 20)
    ax1.set_title('Window Mean Trends - Accelerometer')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # FFT plot
    line_fft_x, = ax2.plot([], [], 'r-', label='X', linewidth=1.5)
    line_fft_y, = ax2.plot([], [], 'g-', label='Y', linewidth=1.5)
    line_fft_z, = ax2.plot([], [], 'b-', label='Z', linewidth=1.5)
    ax2.set_xlim(0, SAMPLE_RATE / 2)  # Nyquist frequency
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT - Current Window Accelerometer')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    lines = {
        'ax': line_ax, 'ay': line_ay, 'az': line_az,
        'fft_x': line_fft_x, 'fft_y': line_fft_y, 'fft_z': line_fft_z
    }
    
    return fig, ax1, ax2, lines

def animate_window1(frame, ax1, ax2, lines):
    """Animation function for window 1 - current window data"""
    window_data, timestamps = plot_data.get_current_window()
    
    if window_data is None:
        return lines.values()
    
    # Update accelerometer lines
    lines['ax'].set_data(timestamps, window_data[:, 0])
    lines['ay'].set_data(timestamps, window_data[:, 1])
    lines['az'].set_data(timestamps, window_data[:, 2])
    
    # Update gyroscope lines
    lines['gx'].set_data(timestamps, window_data[:, 3])
    lines['gy'].set_data(timestamps, window_data[:, 4])
    lines['gz'].set_data(timestamps, window_data[:, 5])
    
    return lines.values()

def animate_window2(frame, ax1, ax2, lines):
    """Animation function for window 2 - means and FFT"""
    # Update window means
    means_data = plot_data.get_window_means()
    
    if means_data is not None:
        lines['ax'].set_data(means_data['ids'], means_data['ax'])
        lines['ay'].set_data(means_data['ids'], means_data['ay'])
        lines['az'].set_data(means_data['ids'], means_data['az'])
        
        # Auto-adjust x-axis for means
        if len(means_data['ids']) > 0:
            ax1.set_xlim(means_data['ids'][0], means_data['ids'][-1] + 1)
    
    # Update FFT
    window_data, _ = plot_data.get_current_window()
    
    if window_data is not None:
        # Compute FFT for each accelerometer axis
        fft_x = np.fft.rfft(window_data[:, 0])
        fft_y = np.fft.rfft(window_data[:, 1])
        fft_z = np.fft.rfft(window_data[:, 2])
        
        # Get frequency bins
        freqs = np.fft.rfftfreq(len(window_data), 1/SAMPLE_RATE)
        
        # Compute magnitudes
        mag_x = np.abs(fft_x)
        mag_y = np.abs(fft_y)
        mag_z = np.abs(fft_z)
        
        # Update FFT lines
        lines['fft_x'].set_data(freqs, mag_x)
        lines['fft_y'].set_data(freqs, mag_y)
        lines['fft_z'].set_data(freqs, mag_z)
        
        # Auto-adjust y-axis for FFT
        max_mag = max(np.max(mag_x), np.max(mag_y), np.max(mag_z))
        ax2.set_ylim(0, max_mag * 1.1)
    
    return lines.values()

def start_plots():
    """Start both matplotlib windows in the main thread"""
    # Setup window 1
    fig1, ax1_1, ax2_1, lines1 = setup_plot_window1()
    ani1 = animation.FuncAnimation(
        fig1, animate_window1, 
        fargs=(ax1_1, ax2_1, lines1),
        interval=100,  # 100ms = 10 FPS
        blit=True,
        cache_frame_data=False
    )
    
    # Setup window 2
    fig2, ax1_2, ax2_2, lines2 = setup_plot_window2()
    ani2 = animation.FuncAnimation(
        fig2, animate_window2, 
        fargs=(ax1_2, ax2_2, lines2),
        interval=100,  # 100ms = 10 FPS
        blit=True,
        cache_frame_data=False
    )
    
    plt.show()

async def main():
    # Start matplotlib in a separate thread
    plot_thread = Thread(target=start_plots, daemon=True)
    plot_thread.start()
    
    logger.info("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for d in devices:
        if d.name and "Old Person Life Invader" in d.name:
            target = d
            break

    if not target:
        logger.error("Peripheral not found!")
        return

    async with BleakClient(target.address) as client:
        logger.info(f"Connected to {target.name}")
        logger.info(f"Calibrating sensors... (collecting {CALIBRATION_SAMPLES} samples)")
        logger.info("Please keep device still during calibration")
        
        await client.start_notify(CHAR_UUID, handle_notification)

        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Disconnecting...")
        finally:
            await client.stop_notify(CHAR_UUID)

if __name__ == "__main__":
    asyncio.run(main())