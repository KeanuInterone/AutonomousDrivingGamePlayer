from InputRecorder.input_recorder import InputRecorder
from ComputerInput.mouse_input import MouseInput


def main():

    # CONFIGURATION
    recording_save_location = '/Users/keanuinterone/Projects/AutonomousDrivingGamePlayer/Recordings'
    screen_input_fps = 10
    screen_input_frame_size = 256
    keys_to_track = ['Key.up', 'Key.left', 'Key.right', 'Key.down']


    # ON CLICK
    def on_click(x, y):
        nonlocal recording_center
        nonlocal waiting_for_input

        print(f'Setting recording center to ({x}, {y})')
        recording_center = (x, y)
        waiting_for_input = False

    # ON ENTER PRESSED
    def on_enter_pressed():
        nonlocal waiting_for_input
        waiting_for_input = False

    # ON ESC PRESSED
    def on_esc_pressed():
        nonlocal waiting_for_input
        waiting_for_input = False
    

    # CREATE MOUSE INPUT TO GET RECORDING AREA CENTER
    mouse_input = MouseInput(
        on_click=on_click
        )
    mouse_input.start_listener()
    recording_center = None
    print('Click center of recording area')
    waiting_for_input = True

    # WAIT FOR ON CLICK TO GET CALLED TO SET RECORDING CENTER
    while waiting_for_input:
        pass

    # STOP MOUSE INPUT
    mouse_input.stop_listener()

    # CREATE THE INPUT RECORDER WITH THE RECORDING CENTER
    input_recorder = InputRecorder(
            keys_to_track=keys_to_track,
            screen_input_fps=screen_input_fps,
            screen_input_frame_size=screen_input_frame_size,
            screen_input_frame_center=recording_center,
            enter_key_pressed=on_enter_pressed,
            esc_key_pressed=on_esc_pressed
        )
    print('Hit Enter to record, and esc to stop')
    waiting_for_input = True

    # WAIT FOR USER TO HIT ENTER
    while waiting_for_input:
        pass

    # START THE RECORDER
    print('Starting recorder...')
    input_recorder.start_new_recording()
    print('Recording Started...')
    waiting_for_input = True

    # WAIT FOR USER TO HIT ESC
    while waiting_for_input:
        pass

    # STOP THE RECORDING
    input_recorder.stop_recording()
    print('Recording Finished')
    input_recorder.save_recording(recording_save_location)
    print('Recording saved')
    


if __name__ == "__main__":
    main()