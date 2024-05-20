# Welcome to the Autonomous Game Player Framework!

This code base is meant to be cloned and fitted to your specific use case but should provide you with the framework and components to get you started gathering training data and executing trained models to autonomously play video games! 🎮

## The Inspiration
The idea that inspired me to build this project was that if models can be trained to drive vehicles based off of video data, surely I can build a system that allows me to train a model to play a video game based off of video data of the game being played along with the corresponding actions taken to play the game successfully. Theoretically all that's involved with playing a video game is the player perceiving the video data from the screen and deciding which keys to press and when. As long as we can design a system that can stream video data into a model to predict what actions to take and when, we can programmatically control a player/vehicle in the game to play autonomously. A fully end to end autonomous system.

**Let's break down how this set of python files allows you to do just that.**

## The Methodology
To gather the **end to end data** to train a model to play a video game, we need a framework that will allow us to do a few things. 
1. 🎥 We need to be able to record the screen as you play the game.
2. 🕹️ We need to be able to record the actions you perform on the controller as you play the game. 
3. 🤖 Lastly we need to be able to programmatically invoke actions on the controller in real time to play the game autonomously. 

Fortunately `python` allows us to do these three things.

Although there are a variety of ways to design a system to play a game, this framework is designed to allow you to obtain the end to end data to train a model, and once trained, implement that model so it can play your chosen game autonomously. 

This project is still in its infancy so the current code base has implementations to record and play games that only require arrow keys to play ⬆️ ⬅️ ➡️ ⬇️. However, the system is designed to be built upon so that more modals such as all keys, mouse, and joystick controls can be added as long as they can be recorded and programmatically controlled. 

There are a variety of games that can be played with just the arrow keys ⬆️ ⬅️ ➡️ ⬇️. In my case I built the system playing a free online driving game 🏎️ located here https://www.crazygames.com/game/rally-racer-dirt. But theoretically the current system can be used to obtain data to train models to play any game that only requires the arrow keys to play.

Hopefully I've intrigued your imagination so let's get into how you can make it a reality. ✨

## Data Collection
First you need to collect your training data. Obviously the type of data you collect will determine how you can train a model. This framework was designed for `supervised learning` use cases, specifically `Imitation Learning`. Imitation Learning is simply training a model to perform as closely to the training examples as possible. So in this case you are collecting data of yourself playing a game to train a model to imitate the actions you would take playing the game.

`$python3 record_driving_game.py`

To do this we can run `$python3 record_driving_game.py` in your terminal. This script instructs you to click on the center of your playing area, and hit `enter` to start recording. Clicking on the center of the playing area tells the system where to get its screenshots from. Additionally you can configure the size of the recording window in the configuration section of `record_driving_game.py`, by default it’s set to `256` pixels. The x, y location of your click and the `screen_input_frame_size` are enough to tell the `ScreenInput` class where it should grab its screenshots from. In addition to the location and frame size, a frames per second should be specified at `screen_input_fps`. This will make the `ScreenInput` class take screenshots of the specific area at the desired rate. Once you've played a game session and are finished recording, hit the `esc` button to stop the recording. Once stopped, the script automatically saves the recording file in the recordings folder. 

The recordings produced are `pickle` files. This was a convenience decision because the array of images don’t need to be converted to a video format only to be converted back to an array of images for preprocessing and training. The same goes for the key recording. It was simpler to just throw the recorded arrays into a pickle file so that you can load the pickle files and use the arrays right away without conversion. This also keeps training examples organized because video data and key data are kept together in the same file. The pickle file is just a `dictionary` containing the keys `screen_frames` and `key_frames` each containing their respective array of the recordings. 

**Once you've collected a sufficient number of examples you are ready to start training a model!**

## Model Training

### Data preprocessing:
https://github.com/KeanuInterone/AutonomousDrivingGamePlayer/blob/main/AutonomousGamePlayerDataProcessor.ipynb

### Model Training:
https://github.com/KeanuInterone/AutonomousDrivingGamePlayer/blob/main/AutonoumouseGamePlayerModelTrainer.ipynb

### Checkpoint to tflite Model:
https://github.com/KeanuInterone/AutonomousDrivingGamePlayer/blob/main/CheckpointToModel.ipynb

I won't go into the details of preprocessing and model training in this documentation. I'll leave that for the notebooks themselves. But one important part that needs to be understood if you want to use this framework out of the box is how the model will make its predictions. Hopefully it’s understood that the model will need to continuously make predictions, one after the other to update the state of the controller. Similar to how you are continuously taking in the screen data and deciding when to press on the gas, brake, turn left, turn right and so on. We need to devise a method of feeding the model what the current state of the video game is so that it can predict what actions it should take. In the case of a driving game, it isn’t enough to make a prediction off of the currently displayed image. Sure, we can see from a single image if there is a clear road ahead or if a turn is approaching, but how can the model understand how fast it’s going, or if it’s currently drifting and needs to counter steer. We need to provide the model with more context by feeding it a set number of previous screen frames so that it can make predictions based on what it sees and the differences it perceives between video frames. 

This introduces the idea of `segmentation`. Segmentation is the process of segmenting our recordings into a fixed window size so that we have a homogeneous input to train the model on. Think of it as a sliding window that we slide over our recordings. We take the image frames from inside the window and target the model to predict the last `key state` frame in that window. In my case I used a segment window size of `20`. This means that the model receives 20 frames of image data and attempts to predict the last `key state` of that 20 frames. Another way to think of 20 frames is that 20 frames taken at 10 frames per second is 2 seconds of video data. So the model takes into context the previous 2 seconds of data to predict what action it should take. 

What all this means is that once you’ve trained your model and want to implement it using this framework you can configure the window size that is sent to your model. Specifically in `auto_play_driving_game.py` you configure `model_input_shape` to be the shape of the data that is fed to your model. In the default case it is set to `(20, 256, 256, 3)`. Meaning 20 frames of RGB images with height and width of 256. 

The framework also expects the trained model to be exported as a `tflite` file.  The decision to go with `tflite` was that it significantly reduces the inference time compared to other model formats. It's important that your model is able to make an inference faster than the frames per second that you've specified. Otherwise the system gets bogged down with backed up inferences and performance is lost. When designing your model factors such as `image size`, `segment length`, and `model size`, all affect `inference time` and `accuracy`. So you’ll have to experiment with these factors to find a balance that works for you. 

## Running Inferences 
After you have trained the model and exported it in a tflite format, you can drag it into the `models` directory and specify its path in the `auto_play_driving_game.py` configuration section. You should also configure the same input shape, frame size, and the frames per second that your model was trained for. 

`$python3 auto_play_driving_game.py`

After configuration you can run the script with `$python3 auto_play_driving_game.py` in your terminal. The script instructs you to click in the center of the game area, exactly the same as the recording. And then when you're ready for the model to start playing autonomously, hit the `enter` key and watch the magic! ✨

