from ComputerPlayer.computer_player import ComputerPlayer
from ComputerPlayer.ModelInstances.single_action_model import SingleActionModel
from ComputerInput.key_input import KeyInput
from ComputerInput.mouse_input import MouseInput



def main():

    # CONFIGURATION
    model_path = '/Users/keanuinterone/Projects/AutonomousDrivingGamePlayer/Models/model_5.2.1.1.tflite'
    model_input_shape = (20, 256, 256, 3)
    activation_threshold = 0.5
    normalize_images_for_prediction = True
    print_inference_times = False
    print_prediction = True
    vision_frame_size = 256
    fps=10


    # ON CLICK
    def on_click(x, y):
        nonlocal vision_frame_center
        nonlocal waiting_for_input

        print(f'Setting recording center to ({x}, {y})')
        vision_frame_center = (x, y)
        waiting_for_input = False

    # ON ENTER KEY PRESSED
    def enter_key_pressed():
        nonlocal waiting_for_input
        waiting_for_input = False

    # ON ESC KEY PRESSED
    def esc_key_pressed():
        nonlocal waiting_for_input
        waiting_for_input = False


    # CREATE MOUSE INPUT TO GET VISION FRAME CENTER
    mouse_input = MouseInput(
        on_click=on_click
    )
    mouse_input.start_listener()
    vision_frame_center = None
    print('Click center of recording area')
    waiting_for_input = True

    # WAIT FOR ON CLICK TO GET CALLED TO SET VISION FRAME CENTER
    while waiting_for_input:
        pass

    # STOP MOUSE INPUT
    mouse_input.stop_listener()

    # CREATE KEY INPUT AND LISTEN FOR ENTER AND ESC
    key_input = KeyInput(
        enter_key_pressed=enter_key_pressed,
        exit_key_pressed=esc_key_pressed
    )
    key_input.start_listener()
    print('Hit Enter to start player, Esc to stop')
    waiting_for_input = True

    # WAIT FOR ENTER KEY TO START PLAYER
    while waiting_for_input:
        pass

    # CREATE MODEL
    model = SingleActionModel(
        tflite_file_path=model_path,
        input_shape=model_input_shape,
        activation_threshold=activation_threshold,
        normalize_images=normalize_images_for_prediction,
        print_prediction=print_prediction,
        print_inference_time=print_inference_times
    )

    # CREATE COMPUTER PLAYER
    computer_player = ComputerPlayer(
        model=model,
        vision_frame_size=vision_frame_size,
        vision_frame_center=vision_frame_center,
        vision_frames_per_second=fps
    )

    # START COMPUTER PLAYER
    computer_player.start_player()
    print('Player Started...')

    waiting_for_input = True

    # WAIT FOR USER TO HIT ESC
    while waiting_for_input:
        pass

    # STOP SCREEN INPUT
    computer_player.stop_player()
    print('Player stoped')

if __name__ == "__main__":
    main()
