import asyncio
import logging
from cust_senml_decode import decode_senml_cbor
from cbor2 import loads
from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

# Feather Sense BLE settings
DEVICE_NAME = "Old Person Life Invader"  # Change this to match your device's advertised name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ble = BLERadio()

buffer = bytearray()

def find_feather_sense():
    """Scan for the Feather Sense device"""
    logger.info("Scanning for Feather Sense...")
    
    while True:
        for adv in ble.start_scan(ProvideServicesAdvertisement, timeout=10):
            if (adv.complete_name == DEVICE_NAME) and (UARTService in adv.services):
                logger.info(f"Found device: {adv.complete_name} at {adv.address}")
                ble.stop_scan()
                return ble.connect(adv)

        logger.info("Device not found, retrying scan...")

async def main():
    global buffer
    try:
        device = find_feather_sense()
        uart = device[UARTService]
        while device.connected:
            chunk = uart.read()
            if not chunk:
                continue
            buffer.extend(chunk)

            try:
                while b"\n" in buffer:
                    msg, _, buffer = buffer.partition(b"\n")
                    logger.info(f"Raw data: {msg}")
                    cbor = decode_senml_cbor(msg)
                    logger.info(f"Received: {cbor}")
            except Exception as e:
                print(f"Had an error decoding: {e}")
            # await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("User interrupted. Shutting down...")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
