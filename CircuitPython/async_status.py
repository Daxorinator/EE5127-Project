# Based off of https://learn.adafruit.com/cooperative-multitasking-in-circuitpython-with-asyncio/concurrent-tasks

async def blink(np_pin, interval, count=0, colour=(255, 0, 0)):  # Don't forget the async!
    with neopixel.NeoPixel(board.NEOPIXEL, 1, brightness=0.3, auto_write=True) as pixel: 
        forever = True
        while forever:
            if count != 0:
                forever = False
            for _ in range(count):
                pixel[0] = colour
                await asyncio.sleep(interval)  # Don't forget the await!
                pixel[0] = (0, 0, 0)
                await asyncio.sleep(interval)  # Don't forget the await!

