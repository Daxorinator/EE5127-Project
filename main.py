import asyncio
import struct
import io
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bleak import BleakScanner, BleakClient

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a custom formatter with no prefix
formatter = logging.Formatter('%(message)s')

# Create file handler that overwrites the file
file_handler = logging.FileHandler('data.csv', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class DataWindow:
    def __init__(self, window_size=133):
        self.window_size = window_size
        self.data = []
        self.ready = False
        self.time = np.linspace(0, 2.56, window_size)  # Time axis for 2.56s window

    def add_sample(self, sample):
        self.data.append(sample)
        if len(self.data) >= self.window_size:
            self.ready = True
            return True
        return False

    def get_data(self):
        return self.data[:self.window_size]

    def get_arrays(self):
        if not self.data:
            return None
        data_array = np.array(self.data[:self.window_size])
        return {
            'time': self.time,
            'accel': data_array[:, :3],  # First 3 columns (ax, ay, az)
            'gyro': data_array[:, 3:]    # Last 3 columns (gx, gy, gz)
        }

    def shift_window(self):
        shift_size = self.window_size // 2
        self.data = self.data[shift_size:]
        self.ready = False

class DataCollector:
    def __init__(self):
        self.current_window = DataWindow()
        self.window_count = 0
        self.setup_plot()

    def setup_plot(self):
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Setup accelerometer subplot
        self.ax1.set_title('Accelerometer Data')
        self.ax1.set_ylabel('Acceleration (m/s²)')
        self.ax1.grid(True)
        self.accel_lines = [self.ax1.plot([], [], label=label)[0] 
                           for label in ['X', 'Y', 'Z']]
        self.ax1.legend()

        # Setup gyroscope subplot
        self.ax2.set_title('Gyroscope Data')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Angular Velocity (dps)')
        self.ax2.grid(True)
        self.gyro_lines = [self.ax2.plot([], [], label=label)[0] 
                          for label in ['X', 'Y', 'Z']]
        self.ax2.legend()

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, data_dict):
        if data_dict is None:
            return

        # Update accelerometer data
        for i, line in enumerate(self.accel_lines):
            line.set_data(data_dict['time'], data_dict['accel'][:, i])
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update gyroscope data
        for i, line in enumerate(self.gyro_lines):
            line.set_data(data_dict['time'], data_dict['gyro'][:, i])
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_sample(self, sample):
        if self.current_window.add_sample(sample):
            # Window is full
            window_data = self.current_window.get_data()
            self.process_window(window_data)
            self.current_window.shift_window()
            self.window_count += 1

    def process_window(self, window_data):
        logger.info(f"Window {self.window_count}:")
        for sample in window_data:
            ax, ay, az, gx, gy, gz = sample
            logger.info(f"{ax:.1f},{(ay + 0.2):.1f},{(az - 9.8):.1f},{gx:.1f},{gy:.1f},{gz:.1f}")
        logger.info("")  # Empty line between windows
        
        # Update the plot with the current window data
        data_arrays = self.current_window.get_arrays()
        self.update_plot(data_arrays)

data_collector = DataCollector()

def handle_notification(_, data: bytearray):
    try:
        values = struct.unpack("<ffffff", data)
        data_collector.add_sample(values)
    except Exception as e:
        logger.error(f"Decode error: {e}")

async def main():
    logger.info("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for d in devices:
        # logger.info(f"Found {d.name} ({d.address})")
        if d.name and "Old Person Life Invader" in d.name:
            target = d
            break

    if not target:
        logger.error("Peripheral not found!")
        return

    async with BleakClient(target.address) as client:
        logger.info("acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z")
        await client.start_notify(CHAR_UUID, handle_notification)

        # logger.info("Subscribed to accel+gyro data. Listening (Ctrl+C to quit)...")
        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Disconnecting...")
        finally:
            await client.stop_notify(CHAR_UUID)

if __name__ == "__main__":
    asyncio.run(main())

