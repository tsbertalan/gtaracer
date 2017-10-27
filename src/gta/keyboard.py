import keyboard

from gta.recorders import BaseRecorder, BaseTask

class KeyboardRecorder(BaseRecorder):

    def __init__(self, period=2):
        super(self.__class__, self).__init__(period=None, Task=None)
        self.scanCodes = {}

    def eventCallback(self, event):
        self.resultQueue.put(event)

    def start(self):
        keyboard.hook(self.eventCallback)

    def stop(self):
        keyboard.unhook(self.eventCallback)

    @property
    def resultsList(self):
        while not self.resultQueue.empty():
            event = self.resultQueue.get()
            time = event.time
            scanCode = event.scan_code
            if scanCode not in self.scanCodes:
                self.scanCodes[scanCode] = event.name
            self._resultsList.append((
                time, 
                [
                    event.scan_code, 
                    event.is_keypad, 
                    event.event_type=='down'
                ]
            ))
        return self._resultsList