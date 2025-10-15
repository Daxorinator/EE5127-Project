# Provide a remote sensing service over Bluetooth Low-Energy (BLE).
# ----------------------------------------------------------------
# Import the standard Python time functions.
import time
import digitalio
import asyncio
import board
import cbor2

# Import the Adafruit Bluetooth library.  Technical reference:
# https://circuitpython.readthedocs.io/projects/ble/en/latest/api.html
from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService
from senml.senml_pack import SenmlPack
from senml.senml_record import SenmlRecord
from adafruit_apds9960.apds9960 import APDS9960
from adafruit_bmp280 import Adafruit_BMP280_I2C
from adafruit_sht31d import SHT31D

# ----------------------------------------------------------------
# Initialize global variables for the main loop.
ledpin_blue = digitalio.DigitalInOut(board.BLUE_LED)
ledpin_blue.direction = digitalio.Direction.OUTPUT

ledpin_red = digitalio.DigitalInOut(board.RED_LED)
ledpin_red.direction = digitalio.Direction.OUTPUT


i2c = board.I2C()  # uses board.SCL and board.SDA
ble = BLERadio()
ble.name = "Old Person Life Invader"
uart = UARTService()
advertisement = ProvideServicesAdvertisement(uart)

# check for LSM6DS33 or LSM6DS3TR-C
try:
    from adafruit_lsm6ds.lsm6ds33 import LSM6DS33 as LSM6DS
    lsm6ds = LSM6DS(i2c)
except RuntimeError:
    from adafruit_lsm6ds.lsm6ds3 import LSM6DS3 as LSM6DS
    lsm6ds = LSM6DS(i2c)


# Flags for detecting state changes.
advertised = False
connected  = False

# https://learn.adafruit.com/adafruit-feather-sense/circuitpython-sense-demo

while True:

    if not advertised:
        ble.start_advertising(advertisement)
        print("Waiting for connection.")
        advertised = True
        ledpin_red.value = True
        continue
        
    elif connected and not ble.connected:
        print("Connection lost.")
        connected = False
        advertised = False
        ledpin_red.value = True
        ledpin_blue.value = False # blue led off for Bluetooth disconnect
        continue
    
    elif not connected and ble.connected:
        print("Connection received.")
        connected = True
        ledpin_red.value = False
        ledpin_blue.value = True # blue led on for Bluetooth connect
    
    elif not connected:
        continue 

    # Accelerometer and Gyro
    accel_x = lsm6ds.acceleration[0]
    accel_y = lsm6ds.acceleration[1]
    accel_z = lsm6ds.acceleration[2]
    
    gyro_x = lsm6ds.gyro[0]
    gyro_y = lsm6ds.gyro[1]
    gyro_z = lsm6ds.gyro[2]


    pack = SenmlPack("feathersense")
    pack.base_time = time.time() 

    # Acceleration
    pack.add(SenmlRecord("accel_x", value=accel_x, unit="m/s2"))
    pack.add(SenmlRecord("accel_y", value=accel_y, unit="m/s2"))
    pack.add(SenmlRecord("accel_z", value=accel_z, unit="m/s2"))

    pack.add(SenmlRecord("gyro_x", value=gyro_x, unit="deg/s"))
    pack.add(SenmlRecord("gyro_y", value=gyro_y, unit="deg/s"))
    pack.add(SenmlRecord("gyro_z", value=gyro_z, unit="deg/s"))

    # payload = pack.to_json().encode("utf-8")
    payload = pack.to_cbor()
    print(payload)
    
    if connected:
        uart.write(payload + b"\n")
    time.sleep(0.1)
