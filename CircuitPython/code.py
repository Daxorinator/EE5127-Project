from adafruit_ble.services import Service
from adafruit_ble.characteristics import Characteristic
from adafruit_ble.uuid import VendorUUID
import time
from cbor2 import dumps
import board
import struct

from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService
# from senml.senml_pack import SenmlPack
# from senml.senml_record import SenmlRecord
from adafruit_apds9960.apds9960 import APDS9960
from adafruit_bmp280 import Adafruit_BMP280_I2C
from adafruit_sht31d import SHT31D


class SensorService(Service):
    uuid = VendorUUID("12345678-1234-5678-1234-56789abcdef0")  # random UUID
    sensor_data = Characteristic(
        uuid=VendorUUID("12345678-1234-5678-1234-56789abcdef1"),
        properties=Characteristic.NOTIFY,
        max_length=128
    )

i2c = board.I2C()  # uses board.SCL and board.SDA
ble = BLERadio()
ble.name = "Old Person Life Betterer"
sensor_service = SensorService()
advertisement = ProvideServicesAdvertisement(sensor_service)

# check for LSM6DS33 or LSM6DS3TR-C
try:
    from adafruit_lsm6ds.lsm6ds33 import LSM6DS33 as LSM6DS
    lsm6ds = LSM6DS(i2c)
except RuntimeError:
    from adafruit_lsm6ds.lsm6ds3 import LSM6DS3 as LSM6DS
    lsm6ds = LSM6DS(i2c)


target_dt = 1.0/50.0

while True:
    print("Starting the old person life betterer")
    ble.start_advertising(advertisement)
    while not ble.connected:
        pass

    while ble.connected:
        t0 = time.monotonic()

        # Read accel x
        accel_x = lsm6ds.acceleration[0]  # just X axis

        # Convert float -> 4-byte little-endian binary
        payload = struct.pack("<f", accel_x)

        try:
            sensor_service.sensor_data = payload
        except Exception:
            pass

        # maintain ~50 Hz
        dt = time.monotonic() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)
