from typing import Callable, Optional
import numpy as np

class ModelInterface:
    def __init__(self):
        """
        Constructor for the Model class.

        Args:
        - on_prediction_made (function): A callback function that will be called when the model makes a prediction.
        """
        self.on_prediction_made: Optional[Callable[[np.ndarray], None]] = None

    def feed_image(self,
                   image: np.ndarray) -> None:
        """
        Function to be called to provide the model with image data.

        Args:
        - image: The input image to be processed by the model.
        """
        raise NotImplementedError("Subclasses must implement the on_image_input method.")
