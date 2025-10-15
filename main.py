import asyncio
import struct
from bleak import BleakScanner, BleakClient

# UUIDs must match the ones you defined in your peripheral code
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

def handle_notification(_, data: bytearray):
    """Callback for received accel_x float packets"""
    try:
        # Unpack little-endian float
        (accel_x,) = struct.unpack("<f", data)
        print(f"Accel X: {accel_x:.3f} m/s^2")
    except Exception as e:
        print(f"Decode error: {e}")

async def main():
    # Scan for your device
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for d in devices:
        print(f"Found {d.name} ({d.address})")
        if d.name and "Old Person Life Betterer" in d.name:
            target = d
            break

    if not target:
        print("Peripheral not found!")
        return

    # Connect and subscribe
    async with BleakClient(target.address) as client:
        print("Connected to", target.name)

        # Subscribe to notifications
        await client.start_notify(CHAR_UUID, handle_notification)

        print("Subscribed to accel_x data. Listening (Ctrl+C to quit)...")
        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("Disconnecting...")
        finally:
            await client.stop_notify(CHAR_UUID)

if __name__ == "__main__":
    asyncio.run(main())
