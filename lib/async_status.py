import neopixel
import board
import asyncio

class LEDController:
    def __init__(self):
        self.current_color = (255, 0, 0)  # Default red
        self.pixel = neopixel.NeoPixel(board.NEOPIXEL, 1, brightness=0.3, auto_write=True)
        self.running = True

    def set_color(self, color):
        """Change the LED color. Color should be a tuple of (R, G, B)"""
        self.current_color = color

    def stop(self):
        """Stop the blinking animation"""
        self.running = False
        self.pixel[0] = (0, 0, 0)

    async def blink(self, interval=0.5):
        """Blink the LED with the current color at the specified interval"""
        self.running = True
        while self.running:
            self.pixel[0] = self.current_color
            await asyncio.sleep(interval)
            self.pixel[0] = (0, 0, 0)
            await asyncio.sleep(interval)