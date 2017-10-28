import keyboard

from gta.recording import BaseRecorder
import gta.eventIDs

class KeyboardRecorder(BaseRecorder):

    def __init__(self):
        super(self.__class__, self).__init__(period=None, Task=None)

    def eventCallback(self, event):
        self.resultsQueue.put(event)

    def start(self):
        keyboard.hook(self.eventCallback)

    def stop(self):
        keyboard.unhook(self.eventCallback)

    @property
    def resultsList(self):
        while not self.resultsQueue.empty():
            event = self.resultsQueue.get()
            time = event.time
            key = event.name
            if event.is_keypad:
                key += '_keypad'
            eventID = gta.eventIDs.keys2eids.get(key, -1)
            if eventID == -1:
                from warnings import warn
                warn("Didn't find '%s' in eventIDs table; using %d." % (key, eventID))
            self._resultsList.append((
                time, 
                [
                    eventID,
                    event.event_type=='down'
                ]
            ))
        return self._resultsList