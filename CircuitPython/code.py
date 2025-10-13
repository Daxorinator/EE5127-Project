# Provide a remote sensing service over Bluetooth Low-Energy (BLE).
# ----------------------------------------------------------------
# Import the standard Python time functions.
import time
import digitalio
import board

# Import the Adafruit Bluetooth library.  Technical reference:
# https://circuitpython.readthedocs.io/projects/ble/en/latest/api.html
from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

# ----------------------------------------------------------------
# Initialize global variables for the main loop.
ledpin = digitalio.DigitalInOut(board.BLUE_LED)
ledpin.direction = digitalio.Direction.OUTPUT

i2c = board.I2C()  # uses board.SCL and board.SDA
ble = BLERadio()
ble.name = "Seán Kelly"
uart = UARTService()
advertisement = ProvideServicesAdvertisement(uart)


try:
    from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
    sensor = LSM6DS(i2c)
except RuntimeError:
    from adafruit_lsm6ds.lsm6ds33 import LSM6DS33 as LSM6DS
    sensor = LSM6DS(i2c)

# Flags for detecting state changes.
advertised = False
connected  = False

# The sensor sampling rate is precisely regulated using the following timer variables.
sampling_timer    = 0.0
last_time         = time.monotonic()
sampling_interval = 0.10

# ----------------------------------------------------------------
# Begin the main processing loop.

while True:

    # Read the accelerometer at regular intervals.  Measure elapsed time and
    # wait until the update timer has elapsed.
    now = time.monotonic()
    interval = now - last_time
    last_time = now
    sampling_timer -= interval
    if sampling_timer < 0.0:
        sampling_timer += sampling_interval
        accel_x, accel_y, accel_z = sensor.acceleration  # m/s^2
        gyro_x, gyro_y, gyro_z = sensor.gyro            # dps (degrees per second)

        #print(f"Gyro:  {gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f} dps")
    else:
        x = None

    if not advertised:
        ble.start_advertising(advertisement)
        print("Waiting for connection.")
        advertised = True

    if not connected and ble.connected:
        print("Connection received.")
        connected = True
        ledpin.value = True
        
    if connected:
        if not ble.connected:
            print("Connection lost.")
            connected = False
            advertised = False
            ledpin.value = False            
        else:
            if accel_x is not None:
                uart.write(b"Accel: %.3f,%.3f,%.3f | Temp: %.3f,%.3f,%.3f\n" % (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z))
