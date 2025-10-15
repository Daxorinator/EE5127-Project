import struct
import time
from async_status import LEDController
import board
import asyncio

from adafruit_ble import BLERadio
from adafruit_lsm6ds import Rate
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services import Service
from adafruit_ble.characteristics import Characteristic
from adafruit_ble.uuid import VendorUUID

# Custom BLE Service
class SensorService(Service):
    uuid = VendorUUID("12345678-1234-5678-1234-56789abcdef0")
    sensor_data = Characteristic(
        uuid=VendorUUID("12345678-1234-5678-1234-56789abcdef1"),
        properties=Characteristic.NOTIFY,
        max_length=24  # 6 floats * 4 bytes each
    )

i2c = board.I2C()

try:
    from adafruit_lsm6ds.lsm6ds33 import LSM6DS33 as LSM6DS
    lsm6ds = LSM6DS(i2c)
except RuntimeError:
    from adafruit_lsm6ds.lsm6ds3 import LSM6DS3 as LSM6DS
    lsm6ds = LSM6DS(i2c)

lsm6ds.accelerometer_data_rate = Rate.RATE_52_HZ
lsm6ds.gyro_data_rate = Rate.RATE_52_HZ

led = LEDController()
blink_task = asyncio.create_task(led.blink())


ble = BLERadio()
ble.name = "Old Person Life Betterer"
sensor_service = SensorService()
advertisement = ProvideServicesAdvertisement(sensor_service)

target_dt = 1.0 / 50.0  # 50 Hz

while True:
    print("Starting the old person life betterer")
    ble.start_advertising(advertisement)
    while not ble.connected:
        led.set_color((255, 0, 0))

    while ble.connected:
        led.set_color(0, 255, 0)
        t0 = time.monotonic()

        # Read accel + gyro
        accel = lsm6ds.acceleration   # (x,y,z)
        gyro  = lsm6ds.gyro           # (x,y,z)

        # Pack all 6 floats: ax, ay, az, gx, gy, gz
        payload = struct.pack("<ffffff",
                              accel[0], accel[1], accel[2],
                              gyro[0],  gyro[1],  gyro[2])

        try:
            sensor_service.sensor_data = payload
        except Exception:
            pass

        # maintain ~50 Hz
        dt = time.monotonic() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)
