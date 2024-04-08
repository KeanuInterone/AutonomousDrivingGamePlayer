import numpy as np
import pyautogui
import threading
import cv2

################
# SCREEN INPUT #
################

# CLASS
class ScreenInput:
    """
    A class for listening to screen input
    """

    # INIT
    def __init__(
            self,
            frame_size=256,
            frame_center=(128, 128),
            fps=60,
            on_screenshot=None
            ):
        """
        Initializes the Screen Input class that can recive screen input from the desired 
        location

        Args:
            frame_size (int, optional): Size of one side of a square frame
            frame_center (tuple (x, y), optional): Center of the square frame
            fps (int, optional): Frames per second to grab screen input
            on_screenshot (function(img), optional): Callback when screenshot is taken
        """
        self.frame_size = frame_size
        self.frame_center = frame_center
        self.fps = fps
        self.on_screenshot = on_screenshot
        self.is_listening = False

        # Get frame coordinates
        self.left = int(self.frame_center[0] - (frame_size / 2))
        self.top = int(self.frame_center[1] - (frame_size / 2))

    # START LISTENER
    def start_listener(self):
        """
        Starts screen listening. Just returns if already listening to screen 
        """
        if self.is_listening:
            return
        self.is_listening = True
        self.schedule_grab_screenshot()

    # STOP LISTENER
    def stop_listener(self):
        """
        Stops screen listening
        """
        self.is_listening = False

    # SCHEDULE GRAB SCREENSHOT
    def schedule_grab_screenshot(self):
        """
        Called to start calling grab_screen shots at the set fps.
        Stops when is_listening is set to False
        """
        if not self.is_listening:
            return
        threading.Timer(1.0 / self.fps, self.schedule_grab_screenshot).start()
        self.grab_screenshot()
    
    # GRAB SCREENSHOT
    def grab_screenshot(self):
        """
        Uses pyautogui to take screenshot of the specified screen area
        definded by frame_size and frame_center. Then calls on_screenshot
        with the image
        """
        if self.on_screenshot:
            img = pyautogui.screenshot(
                region=(
                    self.left, 
                    self.top, 
                    self.frame_size, 
                    self.frame_size
                )
            )
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            self.on_screenshot(img)





