from collections import deque
import time

import xinput

from gta.recorders import BaseRecorder, BaseTask

# x360ce.ini assigns int or str keys to each button press.
eventNames = {
    1: 'd_up', 2: 'd_down', 3: 'd_left', 4: 'd_right',
    5: 'start', 6: 'back',
    7: 'l_stick', 8: 'r_stick',
    9: 'l_bumper', 10: 'r_bumper',
    13: 'a', 14: 'b',
    15: 'x', 16: 'y',
}
buttonNames = list(eventNames.keys())
eventKeys = []
eventKeys.extend(buttonNames)
axisNames = [
    'l_thumb_x',
    'l_thumb_y',
    'r_thumb_x',
    'r_thumb_y',
    'left_trigger',
    'right_trigger',
]
eventKeys.extend(axisNames)
eventKeys = tuple(eventKeys)
for k in eventKeys:
    if isinstance(k, str):
        eventNames[k] = k

# Assign arbitrary eventIDs for featurizing later.
eventIDs = {
    k: i for (i, k) in enumerate(eventKeys)
}

def eid2key(eid):
    for eventKey, otherId in eventIDs.items():
        if otherId == eid:
            return eventKey

def eid2name(eid):
    return eventNames[eid2key(eid)]

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
                time, (eventIDs[eventKey], eventValue)
            ))
        return self._resultsList
