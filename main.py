import asyncio
import struct
import io
import logging
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

def handle_notification(_, data: bytearray):
    try:
        # print(f"Accel (m/s^2): x={ax:.2f}, y={ay:.2f}, z={az:.2f} | "
        #       f"Gyro (dps): x={gx:.2f}, y={gy:.2f}, z={gz:.2f}")
        ax, ay, az, gx, gy, gz = struct.unpack("<ffffff", data)
        logger.info(f"{ax:.1f},{(ay + 0.2):.1f},{(az - 9.8):.1f},{gx:.1f},{gy:.1f},{gz:.1f}")
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

