import asyncio
import struct
import io
import logging
from bleak import BleakScanner, BleakClient
from collections import deque
import statistics

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

# Calibration settings
CALIBRATION_SAMPLES = 50  # Number of samples to average for calibration
NOISE_THRESHOLD = 0.1     # Values below this are considered noise (m/s^2 or dps)

# Windowing settings
WINDOW_SIZE = 128         # 2.56 seconds at 50 Hz
WINDOW_STEP = 64          # 50% overlap (step by half the window size)

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
        # For Z-axis accelerometer, offset to remove gravity (9.8 m/s^2)
        self.az_offset = statistics.mean(az_samples) - 9.8
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
        
        # If we have a complete window, log it
        if window is not None:
            window_id = (len(sensor_processor.window_buffer) - WINDOW_SIZE) // WINDOW_STEP + 1
            for sample_id, (ax, ay, az, gx, gy, gz) in enumerate(window):
                logger.info(f"{window_id},{sample_id},{ax:.1f},{ay:.1f},{az:.1f},{gx:.1f},{gy:.1f},{gz:.1f}")
    except Exception as e:
        logger.error(f"Decode error: {e}")

async def main():
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