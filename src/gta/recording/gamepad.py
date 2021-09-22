from collections import deque
import numpy as np
import time

import xinput

from gta.recording import BaseRecorder, BaseTask
import gta.eventIDs

import multiprocessing
import multiprocessing.queues

from collections import deque


class _GamepadListener(object):

    def __init__(self):
        try:
            self.joystick = xinput.XInputJoystick.enumerate_devices()[0]
        except IndexError:
            raise IndexError('Probably no gamepads are connected.')
        self.state = np.zeros((20,))  # probably actually 16 long

        @self.joystick.event
        def on_button(button, pressed):
            self.state[gta.eventIDs.keys2eids[button]] = pressed

        @self.joystick.event
        def on_axis(axis, value):
            self.state[gta.eventIDs.keys2eids[axis]] = value

    def __call__(self):
        self.joystick.dispatch_events()

def _dispatchEvents(resultsQueue, period):
    listener = _GamepadListener()
    while True:
        listener()
        resultsQueue.put(np.copy(listener.state))
        try:
            time.sleep(period)
        except KeyboardInterrupt:
            break

class GamepadQuery(object):

    def __init__(self, maxOldStates=10, listenPeriod=.0001):
        self._resultsQueue = multiprocessing.Queue()
        self._transferred = deque(maxlen=maxOldStates)
        self._listenPeriod = listenPeriod
        self.start()

    def start(self):
        self.worker = multiprocessing.Process(target=_dispatchEvents, args=(self._resultsQueue, self._listenPeriod))
        self.worker.daemon = True
        self.worker.start()

    def stop(self):
        if hasattr(self, 'worker'):
            self.worker.terminate()

    def __del__(self):
        self.stop()

    def _transfer(self):
        while True:
            try:
                self._transferred.append(self._resultsQueue.get_nowait())
            except multiprocessing.queues.Empty:
                break

    @property
    def state(self):
        self._transfer()
        if len(self) == 0:
            return np.zeros((20,))
        else:
            return np.copy(self._transferred[-1])

    def __len__(self):
        self._transfer()
        return len(self._transferred)

    def __getitem__(self, index):
        return self._transferred[index]

    # def save(self, fpath):
    #     self._transfer()
    #     np.save(fpath, np.stack([self._transferred.popleft() for _ in range(len(self._transferred))]))


class GamepadTask(BaseTask):

    def __init__(self, resultsQueue=None):
        self.joystick = xinput.XInputJoystick.enumerate_devices()[0]
        self.events = []
        assert resultsQueue is not None
        self.resultsQueue = resultsQueue

        @self.joystick.event
        def on_button(button, pressed):
            self.resultsQueue.put((time.monotonic(), (button, pressed)))

        @self.joystick.event
        def on_axis(axis, value):
            self.resultsQueue.put((time.monotonic(), (axis, value)))

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
