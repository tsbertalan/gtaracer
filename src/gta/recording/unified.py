import time

import numpy as np

import gta.recording.vision
import gta.recording.keyboard
import gta.recording.gamepad
from gta.recording import BaseRecorder

class UnifiedRecorder(BaseRecorder):

    def __init__(self, visionPeriod=.01, gamepadPeriod=.01, includeKeyboard=True):
        super().__init__(period=min(visionPeriod, gamepadPeriod), Task=None)
        self.xrecorder = gta.recording.vision.VisionRecorder(period=visionPeriod)#, manager=self.manager)
        self.yrecorders = [gta.recording.gamepad.GamepadRecorder(period=gamepadPeriod)]#, manager=self.manager)]
        if includeKeyboard:
            self.yrecorders.append(gta.recording.keyboard.KeyboardRecorder())#manager=self.manager))
        self.includeKeyboard = includeKeyboard
        self.recorders = list(self.yrecorders)
        self.recorders.append(self.xrecorder)
        
    def create_subprocesses(self):
        for recorder in self.recorders:
            recorder.create_subprocesses()

    @property
    def resultsList(self):
        raise NotImplementedError

    @property
    def results(self):
        raise NotImplementedError

    @property
    def times(self):
        raise NotImplementedError

    def start(self):
        for recorder in self.recorders:
            recorder.start()

    def stop(self):
        for recorder in self.recorders:
            recorder.stop()

    def YT(self):
        if len(self.yrecorders) > 1:
            times = [recorder.times for recorder in self.yrecorders]
            results = [recorder.results for recorder in self.yrecorders]    
            orders = [np.argsort(t) for t in times]
            times = [t[order] for (t, order) in zip(times, orders)]
            results = [r[order] for (r, order) in zip(results, orders)]
        else:
            results = self.yrecorders[0].results
            times = self.yrecorders[0].times
        return results, times

    def XT(self):
        return self.xrecorder.results, self.xrecorder.times

    def XTYT(self, sparse=False, stripKeyboardPartIfExcluded=True):

        X = list(self.xrecorder.resultsList)
        Y = [list(recorder.resultsList) for recorder in self.yrecorders]

        return X, Y
        

        start = time.time()

        # Extract events as they happened.
        Y, Ty = self.YT()
        X, Tx = self.XT()

        # Generate a list of either input (image) or output (control) changes.
        # Also record the times of events, so we have something to sort by.
        events = np.hstack([
            np.vstack([Tx, np.zeros_like(Tx)]),
            np.vstack([Ty, np.ones_like(Ty)]),
        ]).T
        order = np.argsort(events[:, 0])
        events = events[order]

        # Make a control state vector to be mutated repeatedly.
        state = np.zeros((len(gta.eventIDs.keys2eids,)))

        # Make the output array.
        YatX = np.zeros((len(Tx), len(gta.eventIDs.keys2eids))).astype(float)

        # Record our locations in both input and output with separate pointers.
        ix = 0; iy = 0

        # Loop over events, either advancing one pointer or the other for each.
        for unused_t, isYChange in events:
            if isYChange:
                # If the event was a state change, mutate the control state.
                eventId, change = Y[iy]
                eid = int(eventId)
                state[eid] = change
                iy += 1
            else:
                # Otherwise, record the current control state.
                YatX[ix, :] = np.copy(state)
                ix += 1

        if stripKeyboardPartIfExcluded and not self.includeKeyboard:
            YatX = YatX[:, gta.eventIDs.gamepadEids]
                
        # These control sequences tend to be very sparse arrays
        # (rarely does one hold down nore than 2 controls repeatedly),
        # so it might be a good idea to store them sparsely.
        if sparse:
            import scipy.sparse
            YatX = scipy.sparse.csr_matrix(YatX)

        print('Generated XYT in %.3g seconds.' % (time.time() - start,))

        return X, YatX, Tx

    def toSave(self):
        X, Y = self.XTYT()
        return dict(X=X, Y=Y)
