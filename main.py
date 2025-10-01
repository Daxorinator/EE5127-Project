import asyncio
from bleak import BleakClient, BleakScanner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feather Sense BLE settings
DEVICE_NAME = "Seán Kelly"  # Change this to match your device's advertised name

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Write characteristic
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notify characteristic

# Notification handler
def notification_handler(sender, data):
    """Handle incoming data from the Feather Sense"""
    try:
        decoded_data = data.decode()
        logger.info(f"Received: {decoded_data}")
    except Exception as e:
        logger.error(f"Error decoding data: {e}")

async def find_feather_sense():
    """Scan for the Feather Sense device"""
    logger.info("Scanning for Feather Sense...")
    
    while True:
        try:
            # Scan for devices
            device = await BleakScanner.find_device_by_filter(
                lambda d, ad: d.name and d.name.lower() == DEVICE_NAME.lower()
            )
            
            if device:
                logger.info(f"Found Feather Sense: {device.address}")
                return device
                
            logger.info("Device not found, retrying...")
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error during scanning: {e}")
            await asyncio.sleep(1)

async def run_ble_client():
    """Main BLE client function"""
    try:
        # Find the Feather Sense device
        device = await find_feather_sense()
        if not device:
            logger.error("Could not find Feather Sense device")
            return

        async with BleakClient(device) as client:
            logger.info(f"Connected to {device.address}")

            # Set up notification handler
            await client.start_notify(UART_TX_CHAR_UUID, notification_handler)
            logger.info("Notification handler setup complete")

            # Main communication loop
            while True:
                try:
                    # Just keep the connection alive
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error during communication: {e}")
                    break

    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        # Ensure we stop scanning if something goes wrong
        await BleakScanner.stop()

async def main():
    """Main entry point"""
    try:
        while True:
            await run_ble_client()
            logger.info("Connection lost. Attempting to reconnect...")
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("User interrupted. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await BleakScanner.stop()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
