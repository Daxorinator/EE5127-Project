import asyncio
import struct
import io
import logging
from bleak import BleakScanner, BleakClient

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

def handle_notification(_, data: bytearray):
    try:
        ax, ay, az, gx, gy, gz = struct.unpack("<ffffff", data)
        print(f"Accel (m/s^2): x={ax:.2f}, y={ay:.2f}, z={az:.2f} | "
              f"Gyro (dps): x={gx:.2f}, y={gy:.2f}, z={gz:.2f}")
    except Exception as e:
        print(f"Decode error: {e}")

async def main():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for d in devices:
        print(f"Found {d.name} ({d.address})")
        if d.name and "Old Person Life Invader" in d.name:
            target = d
            break

    if not target:
        print("Peripheral not found!")
        return

    async with BleakClient(target.address) as client:
        print("Connected to", target.name)
        await client.start_notify(CHAR_UUID, handle_notification)

        print("Subscribed to accel+gyro data. Listening (Ctrl+C to quit)...")
        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("Disconnecting...")
        finally:
            await client.stop_notify(CHAR_UUID)

if __name__ == "__main__":
    asyncio.run(main())

