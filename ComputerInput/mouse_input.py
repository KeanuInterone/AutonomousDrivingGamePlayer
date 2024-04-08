from pynput import mouse

##############
# MOUSE INUT #
##############


# CLASS
class MouseInput:
    """
    A class for listening to the mouse input
    """

    # INITIALIZE
    def __init__(
            self,
            on_click=None
            ):
        """
        Initializes a MouseInput class to listen for mouse events

        ARGS:
            on_click (function(x, y), optional): Callback for when the mouse clicks
        """
        self.on_click = on_click
        self.listener = None
        self.is_listening = False

    # START LISTENER
    def start_listener(self):
        """
        Starts the mouse listener.

        Creates a listener if one hasn't been created.
        If the listener is already running it does nothing.
        """
        # If there isn't a listener...
        if self.listener == None:
            # ... create one
            self.listener = mouse.Listener(
                on_click=self.on_mouse_click
            )
            self.listener.start()
            self.is_listening = True

    # STOP LISTENER
    def stop_listener(self):
        """
        Stops the mouse listener
        """
        if self.listener:
            self.listener.stop()
            self.is_listening = False

    # ON MOUSE CLICK
    def on_mouse_click(self, x, y, button, pressed):
        """
        pynput callback for on click. Calls the class callback
        when the mouse press is released
        """
        if pressed:
            return # Only want to call callback on released

        if self.on_click:
            self.on_click(x, y)            

