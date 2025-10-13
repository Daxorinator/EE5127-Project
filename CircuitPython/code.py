# Provide a remote sensing service over Bluetooth Low-Energy (BLE).
# ----------------------------------------------------------------
# Import the standard Python time functions.
import time
import digitalio
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
ledpin= digitalio.DigitalInOut(board.BLUE_LED)
ledpin.direction = digitalio.Direction.OUTPUT

i2c = board.I2C()  # uses board.SCL and board.SDA
ble = BLERadio()
ble.name = "Indoor/Outdoor Detector"
uart = UARTService()
advertisement = ProvideServicesAdvertisement(uart)

# SHT Humidity
humidity_sense = SHT31D(i2c)
# BMP280 temperature and barometric pressure/altitude
temp_barometer_sense = Adafruit_BMP280_I2C(i2c)
# APDS9960 Proximity, Light, Color, and Gesture Sensor
colour_sense = APDS9960(i2c)


# Flags for detecting state changes.
advertised = False
connected  = False

# The sensor sampling rate is precisely regulated using the following timer variables.
sampling_timer    = 0.0 # seconds
last_time         = time.monotonic() # seconds
sampling_interval = 5 * 60 # 5 minute timer

# ----------------------------------------------------------------
# Begin the main processing loop.



# Set this to sea level pressure in hectoPascals at your location for accurate altitude reading.
temp_barometer_sense.sea_level_pressure = 1013.25 # from https://learn.adafruit.com/adafruit-feather-sense/circuitpython-sense-demo


while True:

    if not advertised:
        ble.start_advertising(advertisement)
        print("Waiting for connection.")
        advertised = True
        continue
        
    elif connected and not ble.connected:
        print("Connection lost.")
        connected = False
        advertised = False
        ledpin.value = False # blue led off for Bluetooth disconnect
        continue
    
    elif not connected and ble.connected:
        print("Connection received.")
        connected = True
        ledpin.value = True # blue led on for Bluetooth connect


    now = time.monotonic()
    interval = now - last_time
    last_time = now
    sampling_timer -= interval

    if sampling_timer < 0.0:
        sampling_timer += sampling_interval

        # Read sensors
        temperature = temp_barometer_sense.temperature   # °C
        pressure = temp_barometer_sense.pressure         # hPa
        altitude = temp_barometer_sense.altitude         # m
        humidity = humidity_sense.relative_humidity      # %

        colour_sense.enable_color = True
        colour_sense.enable_proximity = True
        r, g, b, c = colour_sense.color_data
        proximity = colour_sense.proximity
        colour_sense.enable_color = False
        colour_sense.enable_proximity = False

        # approximate lux
        ambient_lux = (0.299 * r + 0.587 * g + 0.114 * b)  # lux

        pack = SenmlPack("feathersense")
        pack.base_time = time.time() 

        # Temperature
        rec_temp = SenmlRecord("temperature")
        rec_temp.value = temperature
        rec_temp.unit = "Cel"
        pack.add(rec_temp)

        # Pressure
        rec_pres = SenmlRecord("pressure")
        rec_pres.value = pressure
        rec_pres.unit = "hPa"
        pack.add(rec_pres)

        # Altitude
        rec_alt = SenmlRecord("altitude")
        rec_alt.value = altitude
        rec_alt.unit = "m"
        pack.add(rec_alt)

        # Humidity
        rec_hum = SenmlRecord("humidity")
        rec_hum.value = humidity
        rec_hum.unit = "%RH"
        pack.add(rec_hum)

        # Light / colour
        rec_lux = SenmlRecord("light_lux")
        rec_lux.value = ambient_lux
        rec_lux.unit = "lux"
        pack.add(rec_lux)

        rec_r = SenmlRecord("color_r")
        rec_r.value = r
        pack.add(rec_r)

        rec_g = SenmlRecord("color_g")
        rec_g.value = g
        pack.add(rec_g)

        rec_b = SenmlRecord("color_b")
        rec_b.value = b
        pack.add(rec_b)

        rec_c = SenmlRecord("color_c")
        rec_c.value = c
        pack.add(rec_c)

        # payload = pack.to_json().encode("utf-8")
        payload = pack.to_cbor()
        print(payload)
        
        if connected:
            uart.write(payload + b"\n")
