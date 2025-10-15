import asyncio, io, logging
from cbor2 import loads, CBORDecoder, CBORDecodeError
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

def decode_stream(stream, callback):

    buffer = io.BytesIO()
    decoder = CBORDecoder(buffer)

    for chunk in stream:
        pos = buffer.tell() # save current buffer position
        buffer.seek(0, io.SEEK_END) # move to end of buffer
        buffer.write(chunk) # write new chunk
        buffer.seek(pos) # restore position

        while True: # The buffer could contain multiple CBOR objects so keep going until there's an error
            try:
                obj = decoder.decode()
                callback(obj)
            except CBORDecodeError:
                break # Not enough data to decode, wait for more
        
        # Compact the buffer
        remaining = buffer.read()
        buffer.seek(0)
        buffer.truncate(0)
        buffer.write(remaining)
        buffer.seek(0)

async def main():
    try:
        device = find_feather_sense()
        uart = device[UARTService]
        while device.connected:
            data = uart.read(64)
            decode_stream(data, lambda obj: logger.info(f"Received: {obj}"))

    except KeyboardInterrupt:
        logger.info("User interrupted. Shutting down...")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
