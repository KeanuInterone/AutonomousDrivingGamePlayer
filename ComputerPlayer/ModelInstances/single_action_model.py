from ..model_interface import ModelInterface
import tensorflow as tf
import numpy as np
import time
from typing import Tuple

class SingleActionModel(ModelInterface):

    # INITIALIZE
    def __init__(
            self, 
            tflite_file_path: str,
            input_shape: Tuple[int, int, int, int],
            activation_threshold: float,
            normalize_images: bool,
            print_prediction=False,
            print_inference_time=False):
        """
        Initializes a Single Action Model. The single action model takes images through feed_image and updates 
        a rolling image frame array of the previous fed images. Uses this set of image frames to make a prediction.
        Every call to feed_image will produce a prediction passed to the on_prediction_made call back.

        Args:
            tflite_file_path: (string) location of the tflite model to load in
            input_shape: (tuple) Frames x Height x Width x Channel shape that the model is expecting
            activation_threshold: (float) Value 0 to 1 of when the raw prediction should be considered "on"
            normalize_images: (bool) Whether images fed into the model should be scalled between 0 and 1
            print_inference_time: (bool optional) Whether to print inference times
        """
        super().__init__()
        self.image_frames = np.zeros(input_shape, dtype=np.float32)
        self.activation_threshold = activation_threshold
        self.normalize_images = normalize_images
        self.inturpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model(tflite_file_path)
        self.print_prediction = print_prediction
        self.print_inference_time = print_inference_time


    # LOAD MODEL
    def _load_model(
            self,
            tflite_file_path: str) -> None:
        """
        INTERNAL CLASS METHOD
        Uses the tflite file path to load interpreter and set input and output details for making
        predictions

        Args:
            tflite_file_path: (string) location of tflite model to load in
        """
        # Load the model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    # PREPROCESS IMAGE
    def _preprocess_image(
            self,
            image: np.ndarray) -> np.ndarray:
        """
        INTERNAL CLASS METHOD
        Houses logic to preprocess raw images

        Args:
            image: (np array) raw image from feed_image
        
        Returns:
            image: (np array) preprocessed image
        """
        # Convert to numpy
        image = np.array(image, dtype=np.float32)

        # If normalize
        if self.normalize_images:
            image = image / 255.0

        return image


    # UPDATE STATE WITH IMAGE
    def _update_state_with_image(
            self,
            image: np.ndarray) -> None:
        """
        INTERNAL CLASS METHOD
        Houses logic to add preprocessed image to image frames. Rolls the images back a frame and 
        adds new image to the end

        Args:
            image: (np array) preprocessed image
        """
        # Shift and place new image at the end of image_frames
        self.image_frames = np.roll(self.image_frames, -1, axis=0)
        self.image_frames[-1] = image


    # MAKE INFERENCE ON STATE
    def _make_inference_on_state(self) -> np.ndarray:
        """
        INTERNAL CLASS METHOD
        Houses logic to run the inference on the current image_frames state

        Returns:
            raw_pred: (np array) raw prediction made by the model. Possibly None if there was an error. 
                      Should be checked by parent function. 
        """
        # If printing inference time...
        start_time = None
        if (self.print_inference_time):
            # ... Start the timer
            start_time = time.time()

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], [self.image_frames])
        self.interpreter.invoke()
        raw_pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # If printing inference time...
        if (self.print_inference_time):
            # ... Stop the timer and print
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Inference Time : {elapsed_time:.4f} seconds")

        return raw_pred
    

    # APPLY THRESHOLD TO RAW PREDICTION
    def _apply_threshold_to_raw_pred(
            self,
            raw_pred: np.ndarray) -> np.ndarray:
        """
        INTERNAL CLASS METHOD
        Applies threshold to raw prediction

        Args:
            raw_pred: (np array) raw prediction

        Returns:
            pred: (np array) Array of 1s or 0s depending on if the input was above threshold value
        """
        return (raw_pred > self.activation_threshold).astype(np.float32)


    # OVERRIDE OF FEED IMAGE 
    def feed_image(
            self,
            image: np.ndarray) -> None:
        # Preprocess
        image = self._preprocess_image(image)

        # Update state with image
        self._update_state_with_image(image)

        # Make inference
        try:
            raw_pred = self._make_inference_on_state()
        except:
            # Sometimes when making lots of inferences quickly (high FPS) it can overlap and cause an error
            # In this case just return 
            return
        
        # Apply threshold
        pred = self._apply_threshold_to_raw_pred(raw_pred)

        # Print prediction
        if self.print_prediction:
            print(pred)

        # call on prediction made
        if self.on_prediction_made is not None:
            self.on_prediction_made(pred)