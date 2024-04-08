from ComputerInput.key_input import KeyInput
from ComputerInput.screen_input import ScreenInput
import pickle
import os
import time
import numpy as np

##################
# INPUT RECORDER #
##################


# CLASS
class InputRecorder:
    """
    A class to initialize different recording types such as keys and screen
    """

    # INITIALIZE
    def __init__(
            self,
            keys_to_track=['Key.up', 'Key.left', 'Key.right', 'Key.down'],
            screen_input_frame_size=256,
            screen_input_frame_center=(128, 128),
            screen_input_fps=60,
            enter_key_pressed=None,
            esc_key_pressed=None
            ):
        """
        Initializer for the Input Recorder class that gives you methods to start and stop recording
        of your key strokes and screen

        Args:
            keys_to_track (array(string), options): Defaults to up, left, right, down keys
            screen_input_frame_size (int, optional): Size of one side of a square frame
            screen_input_frame_center (tuple (x, y), optional): Center of the square frame
            screen_input_fps (int, optional): Frames per second of recording
            enter_key_pressed (function, optional): Function that is called when enter key is pressed
            esc_key_pressed (function, optional): Function that is called when esc key is pressed
        """

        # Key input
        self.key_input = KeyInput(
            key_pressed=self.key_pressed,
            key_released=self.key_released,
            enter_key_pressed=enter_key_pressed,
            exit_key_pressed=esc_key_pressed
        )
        # Start to imediately start listening for enter and esc keys
        self.key_input.start_listener()

        # Screen Input
        self.screen_input = ScreenInput(
            frame_size=screen_input_frame_size,
            frame_center=screen_input_frame_center,
            fps=screen_input_fps,
            on_screenshot=self.on_screenshot
        )

        # Is recording flag
        self.is_recording = False

        # Dict for key states 
        self.key_state = {key: 0 for key in keys_to_track}

        # Recording logs
        self.key_frames = []
        self.screen_frames = []


    # START NEW RECORDING
    def start_new_recording(self):
        """
        Resets current recorded events and video frames and starts a new recording.
        If
        """
        # Reset logs
        self.key_events = []
        self.key_frames = []
        self.screen_frames = []

        # Start the screen listener - start_listener just returns if already listening
        self.screen_input.start_listener()
            
        # Set is recording to true
        self.is_recording = True


    # STOP RECORDING
    def stop_recording(self):
        """
        Stops recording
        """
        # Stop the screen listener
        self.screen_input.stop_listener()
            
        # Flip recording flag
        self.is_recording = False


    # ON KEY PRESSED
    def key_pressed(self, key):
        """
        Callback for KeyInput, called whenever a key is pressed. Changes the state
        of keys pressed. Returns if not recording.
        """
        # If we are not recording...
        if not self.is_recording:
            # ... don't change anything
            return
        
        # Set key to pressed in key state
        if key in self.key_state:
            self.key_state[key] = 1
        
            

    # ON KEY RELEASED
    def key_released(self, key):
        """
         Callback for KeyInput, called whenever a key is released. Changes the state
        of keys pressed. Returns if not recording.
        """
        # If we are not recording...
        if not self.is_recording:
            # ... don't change anything
            return
        
        # Set key to released in key state
        if key in self.key_state:
            self.key_state[key] = 0


    # ON SCREENSHOT
    def on_screenshot(self, img):
        """
        Callback for ScreenInput. Called frames per second every second when ScreenInput
        is listening. Logs the image. Also logs the key state. Returns if not recording
        or recording screen

        ARGS:
            img (np.array): image of the specified area of the screen
        """
        # If we are not recording...
        if not self.is_recording:
            # ... don't log anything
            return
        
        # Add screenshot and keystate logs
        self.screen_frames.append(img)
        self.key_frames.append(list(self.key_state.values()))

    
    # ON SAVE
    def save_recording(self, location):
        """
        Saves input from either key or screen input if specified from initialization.
        Saves as pkl files

        ARGS:
            location (string): directory to save recording bundle
        """
        # Create recording object
        recording = {}
        recording['key_frames'] = np.array(self.key_frames)
        recording['screen_frames'] = np.array(self.screen_frames)
        
        # Create the file name
        timestamp = int(time.time())
        filename = f"recording_{timestamp}.pickle"
        recording_path = os.path.join(location, filename)

        # Save recording to pickel file
        with open(recording_path, 'wb') as handle:
            pickle.dump(recording, handle, protocol=pickle.HIGHEST_PROTOCOL)
