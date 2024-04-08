from pynput import keyboard


##############
# KEY INPUT  #
##############


# CLASS
class KeyInput:
    """
    A class for listening to keyboard input using pynput.

    Attributes:
        key_pressed (function): A callback function to handle key press events.
        key_released (function): A callback function to handle key release events.
        exit_key_pressed (function): A callback function to handle the 'escape' key press event.
        enter_key_pressed (function): A callback function to handle the 'enter' key press event.
        listener_started (function): A callback function to handle the start of the listener.
        listener (pynput.keyboard.Listener): The keyboard listener instance.
        is_listening (bool): Indicates whether the listener is currently running.
    """

    # INITIALIZE
    def __init__(
            self,
            key_pressed=None,
            key_released=None,
            exit_key_pressed=None,
            enter_key_pressed=None,
            listener_started=None
    ):
        """
        Initializes the KeyInput class.

        Args:
            key_pressed (function, optional): Callback for key press events.
            key_released (function, optional): Callback for key release events.
            exit_key_pressed (function, optional): Callback for 'escape' key press event.
            enter_key_pressed (function, optional): Callback for 'enter' key press event.
            listener_started (function, optional): Callback for listener start event.
        """
        self.key_pressed = key_pressed
        self.key_released = key_released
        self.exit_key_pressed = exit_key_pressed
        self.enter_key_pressed = enter_key_pressed
        self.listener_started = listener_started
        self.listener = None
        self.is_listening = False

    # START LISTENER
    def start_listener(self):
        """
        Starts the keyboard listener.

        Creates a listener if one hasn't been created
        If a listener is already running it does nothing.
        """
        # If there isn't a listener...
        if self.listener == None:
            # ...create one
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )

        # If the listener is already listening...
        if self.listener.running:
            # ... dont start it agian
            return

        # Start Listener
        self.listener.start()
        self.is_listening = True

        # Call listener started call back
        if self.listener_started:
            self.listener_started()

    # STOP LISTENER
    def stop_listener(self):
        """
        Stops the keyboard listener if it exists.
        """
        # If there isn't a listener...
        if self.listener == None:
            return

        # Stop Listener
        self.listener.stop()
        self.is_listening = False

    # IS LISTENER RUNNING?
    def listener_running(self):
        """
        Checks if the listener is currently running.

        Returns:
            bool: True if the listener is running, False otherwise.
        """
        return self.is_listening

    # ON KEY PRESSED
    def on_press(self, key):
        """
        Callback function for key press events.

        Args:
            key: The key that was pressed.
        """
        # Get key string pressed
        key_string = str(key)

        # If it was the exit key...
        if key == keyboard.Key.esc:
            # ...call the exit_key_pressed
            if self.exit_key_pressed:
                self.exit_key_pressed()

        # Call key_pressed callback
        if self.key_pressed:
            self.key_pressed(key_string)

    # ON KEY RELEASED
    def on_release(self, key):
        """
        Callback function for key release events.

        Args:
            key: The key that was released.
        """
        # Get key string
        key_string = str(key)

        # Call key_released callback
        if self.key_released:
            self.key_released(key_string)

        # If it was the enter key...
        if key == keyboard.Key.enter:
            # ...call the enter_key_pressed
            if self.enter_key_pressed:
                self.enter_key_pressed()
