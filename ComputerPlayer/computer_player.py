from .model_interface import ModelInterface
from ComputerInput.screen_input import ScreenInput
from pynput.keyboard import Key, Controller
import numpy as np
from typing import Tuple

class ComputerPlayer:

    # INITIALIZE
    def __init__(
            self,
            model: ModelInterface,
            vision_frame_size: int,
            vision_frame_center: Tuple[float, float],
            vision_frames_per_second: int
            ):
        """
        Initializes a Computer Player. This class combined with the
        provided model will feed images from a screen_input to the
        model for it to predict the next action to take. This class
        is purposed to inturpret the predictions made by the model
        and perform the necisary actions to reach the predicted 
        state. 

        Args:
        - model: (ModelInterface) model to feed images to
        - vision_frame_size: (int) Size of one side of the vision window
        - vision_frame_center: (Tuple[float, float]) Center of the vision window
        - vision_frames_per_second: (int) fps for vision to update
        """
        self.keyboard = Controller()
        self.key_state = [0, 0, 0, 0]
        self.model = model
        self.model.on_prediction_made = self._on_prediction_made
        self.screen_input = ScreenInput(
            frame_size=vision_frame_size,
            frame_center=vision_frame_center,
            fps=vision_frames_per_second,
            on_screenshot=self.model.feed_image
        )
        

    # START PLAYER
    def start_player(self) -> None:
        """
        Starts sending screen input to the model to make predictions
        """
        self.screen_input.start_listener()


    # STOP PLAYER
    def stop_player(self) -> None:
        """
        Stops sending screen input to model
        """
        self.screen_input.stop_listener()


    # ON PREDICTION MADE
    def _on_prediction_made(
            self,
            pred: np.ndarray) -> None:
        """
        INTERNAL CLASS METHOD
        Called by model when a prediction is made

        Args:
        - pred: (np.ndarray) prediction
        """
        # Up
        if pred[0] != self.key_state[0]:
            if pred[0]:
                self.keyboard.press(Key.up)
                self.key_state[0] = 1
            else:
                self.keyboard.release(Key.up)
                self.key_state[0] = 0
        
        # Left
        if pred[1] != self.key_state[1]:
            if pred[1]:
                self.keyboard.press(Key.left)
                self.key_state[1] = 1
            else:
                self.keyboard.release(Key.left)
                self.key_state[1] = 0

        # Right
        if pred[2] != self.key_state[2]:
            if pred[2]:
                self.keyboard.press(Key.right)
                self.key_state[2] = 1
            else:
                self.keyboard.release(Key.right)
                self.key_state[2] = 0

        # Down
        if pred[3] != self.key_state[3]:
            if pred[3]:
                self.keyboard.press(Key.down)
                self.key_state[3] = 1
            else:
                self.keyboard.release(Key.down)
                self.key_state[3] = 0



        