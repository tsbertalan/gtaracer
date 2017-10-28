from collections import deque
import time

import xinput

from gta.recorders import BaseRecorder, BaseTask
import gta.eventIDs

class GamepadTask(BaseTask):

    def __init__(self, resultsQueue=None):
        self.joystick = xinput.XInputJoystick.enumerate_devices()[0]
        self.events = []
        assert resultsQueue is not None
        self.resultsQueue = resultsQueue

        @self.joystick.event
        def on_button(button, pressed):
            self.resultsQueue.put((time.time(), (button, pressed)))

        @self.joystick.event
        def on_axis(axis, value):
            self.resultsQueue.put((time.time(), (axis, value)))

    def __call__(self):
        self.joystick.dispatch_events()


class GamepadRecorder(BaseRecorder):

    def __init__(self, period=.01):
        super(self.__class__, self).__init__(
            period=period, Task=GamepadTask, giveQueueDirectly=True
            )

    @property
    def resultsList(self):
        while not self.resultsQueue.empty():
            time, (eventKey, eventValue) = self.resultsQueue.get()
            self._resultsList.append((
                time, (gta.eventIDs.keys2eids[eventKey], eventValue)
            ))
        return self._resultsList
