from __future__ import print_function
import re, traceback

import numpy as np

import time
import keyboard, xinput, vjoy
from multiprocessing import Process, Queue

import tqdm

import errno    
import os, sys

import platform
if platform.system() == 'Windows':
    import win32gui, win32con, win32com.client


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _recordKeys(queue, **kwargs):
    out = keyboard.record(**kwargs)
    queue.put(out)
    return out

class KeyRecordQueuer(object):

    def __init__(self, queue, special='esc'):
        # self.special = special
        # self.specialFlag = False
        self.queue = queue

    def __call__(self, event):
        # if event.name == self.special:
        #     self.specialFlag
        self.queue.put(event)

    def start(self):
        keyboard.hook(self)

    def stop(self):
        keyboard.unhook(self)

keyNames = ['w', 'a', 's', 'd', 'e', 'esc']


class GTAdata(object):

    def __init__(self, fpath=None):

        self._keyRecordQueue = Queue()
        self.keyRecorder = KeyRecordQueuer(self._keyRecordQueue)
        self.joystickRecorder = JoystickRecorder()
        self.clearHistory()

        if fpath is not None:
            self.load(fpath)      

    def clearHistory(self):
        self._imgs = []
        self._imgTimes = []
        self._keyRecord = []
        #self._joystickEventTimes = []
        self._joystickEvents = self.joystickRecorder.events = []

    @property
    def imgs(self):
        if isinstance(self._imgs, list):
            self._imgs = np.stack(self._imgs)
        return self._imgs

    @property
    def imgTimes(self):
        if isinstance(self._imgTimes, list):
            self._imgTimes = np.stack(self._imgTimes)
        return self._imgTimes

    @property
    def keyRecord(self):
        while True:
            if self._keyRecordQueue.empty():
                break
            else:
                record = self._keyRecordQueue.get()
                self._keyRecord.append([
                    record.event_type == 'down',
                    record.time,
                    self.keyNameToNum(record.name)

                ])
                # recordTypes = np.array([r.event_type for r in record])
                # recordNames = np.array([r.name for r in record])
                # recordTimes = np.array([r.time for r in record])

                # self._keyRecord.append(np.array([
                #     (ty=='down', t, self.keyNameToNum(n))
                #     for (ty, t, n) in zip(
                #         recordTypes,
                #         recordTimes,
                #         recordNames,
                #      )
                # ]))
                    
        return self._keyRecord

    @property
    def keyRecords(self):
        if len(self.keyRecord) == 0:
            return np.empty((0, 3))
        else:
            return np.vstack(self.keyRecord)

    keyNames = keyNames
    def keyNumToKeyName(n):
        if n < len(self.keyNames) and 0 <= n:
            return keyNames[n] # else None
    
    def keyNameToNum(self, name):
        try:
            return self.keyNames.index(name)
        except ValueError:
            return -1

    def save(self, fpath=None, compressed=False):
        start = time.time()
        if fpath is None:
            fpath = os.path.join(
                os.path.expanduser('~'),
                'data',
                'gta%s.npz' % start
                )

        
        gta.mkdir_p(os.path.dirname(fpath))

        tosave = dict(
            imgs=self.imgs,
            imgTimes=self.imgTimes,
            numericRecords=[],
            **{k: np.stack(v) for (k,v) in self.joystickRecorder.getEvents().items()},
        )
        if len(self.keyRecord) > 0:
            toSave['numericRecords'] = np.vstack(self.keyRecord)
        
        print('Saving to %s ... ' % fpath, end=''); sys.stdout.flush()
        if compressed:
            np.savez_compressed(fpath, **tosave)
        else:
            np.savez(fpath, **tosave)
        print('done (elapsed %.0f s).' % (time.time() - start,))

    def load(self, fpath):
        start = time.time()
        print('Loading %s ...' % fpath, end=' '); sys.stdout.flush()
        data = np.load(fpath)
        self._imgTimes = data['imgTimes']
        self._imgs = data['imgs']
        self._keyRecord.append(data['numericRecords'])
        print('done (elapsed %.0f s).' % (time.time() - start,))

    @property
    def states(self):
        if not hasattr(self, '_states'):
            state = [False]*(len(self.keyNames)+1)
            t0 = min(min(self.keyRecords[:, 1]), min(self.imgTimes))
            iStateChange = 0
            iImg = 0
            states = np.ones((len(self.imgTimes), len(state)))*-1
            while True:
                # If we've run out of images to label, stop.
                if iImg >= len(self.imgTimes):
                    break
                else:
                    # What is the next state change?
                    if iStateChange >= len(self.keyRecords):
                        # If the last state change has passed,
                        # and there are more images,
                        # apply the current state to all of them.
                        tStateChange = np.inf
                    else:
                        event, tStateChange, keyNum = self.keyRecords[iStateChange]
                    tImg = self.imgTimes[iImg]
                  
                    # If we've passed the state change, apply it.
                    if tImg > tStateChange:
                        if iStateChange >= len(self.keyRecords):
                            break
                        # Change the state; advance the state change pointer.
                        state[int(keyNum)] = bool(event)
                        event, tStateChange, keyNum = self.keyRecords[iStateChange]
                        iStateChange += 1
                    else:
                        # Save the state and advance the image pointer.
                        states[iImg] = state
                        iImg += 1
            self._states = states
        return self._states


class BoolKeyTrap(object):

    def __init__(self, key='esc', register=True):
        self.key = key
        self.keyHit = False
        if register:
            try:
                keyboard.unhook_key(key)
            except ValueError:
                pass
            keyboard.hook_key(key, keydown_callback=self)

    @property
    def trapped(self):
        return self.keyHit

    def __call__(self, *args, **kwargs):
        self.keyHit = True



def getBar(tqdm_notebook):
    if tqdm_notebook is None:
        bar = lambda x: x
    elif tqdm_notebook:
        bar = tqdm.tqdm_notebook
    else:
        bar = tqdm.tqdm
    return bar

from gta.recording.vision import Window
class GTA(Window, GTAdata):

    def __init__(self, fpath=None):
        wids = []
        def saveWid(wid, *args):
            wids.append(wid)
        win32gui.EnumWindows(saveWid, None)

        widsByTitle = {win32gui.GetWindowText(wid): wid for wid in wids}
        gtaWid = widsByTitle['Grand Theft Auto V']
        
        self.widsByTitle = widsByTitle

        GTAdata.__init__(self, fpath)
        cWindow.__init__(self, gtaWid)

    def startRecordingKeyboard(self):
        self.keyRecorder.start()

    def stopRecordingKeyboard(self):
        self.keyRecorder.stop()

    def grabImagesUntilInterrupted(self, dt=.1, tqdm_notebook=False, limit=np.inf, trapKey='esc'):
        tqdmKwargs = dict(
            desc='Grabbing images', unit='images', unit_scale=False,
            )
        
        if tqdm_notebook is not None:
            print('Making a real bar.')
            bar = getBar(tqdm_notebook)()
        else:
            bar = None

        # Register ESC callback.
        if trapKey is not None:
            escDown = BoolKeyTrap(key=trapKey)
        count = 0
        while True:
            if limit is not np.inf:
                if count >= limit:
                    break
                else:
                    count += 1
            if trapKey is not None:
                if escDown.trapped:
                    break
            try:

                # Get image.
                self._imgTimes.append(time.time())
                self._imgs.append(self.grab())
                if tqdm_notebook is not None:
                    bar.update()
                time.sleep(dt)

                # Get joystick events.
                self.joystickRecorder.listen(np.inf, once=True)

            except KeyboardInterrupt:
                break

    def forward(self):
        win.SetAsForegroundWindow()
        keydown('w')

    def reverse(self):
        win.SetAsForegroundWindow()
        keydown('s')

    def coast(self):
        win.SetAsForegroundWindow()
        keyup('w')
        keyup('s')


import time
class JoystickRecorder(object):
    
    def __init__(self):
        
        self.events = []
        self.joystick = j = xinput.XInputJoystick.enumerate_devices()[0]
        
        def append(thing, value):
            self.events.append((time.time(), thing, value))
        
        @j.event
        def on_button(button, pressed):
            print(button, pressed)
            append(button, pressed)
            
        @j.event
        def on_axis(axis, value):
            append(axis, value)
            
    def listen(self, rate=100, once=False):
        while True:
            try:
                self.joystick.dispatch_events()
                if rate < np.inf:
                    time.sleep(1. / rate)
            except KeyboardInterrupt:
                print('Caught KeyboardInterrupt. Stopping listening.')
                break
            if once:
                break
                
    def getEvents(self):
        tmin = self.events[0][0]
        classes = list(set([x for (t,x,v) in self.events]))
        out = {c: ([], []) for c in classes}
        for (t, x, v) in self.events:
            out[x][0].append(t-tmin)
            out[x][1].append(v)
        return out


class Axis(object):
    
    def __init__(self, vjoyName, inRange=(-1., 1.), outRange=(0, 32000), neutralInValue=0):
        self.vjoyName = vjoyName
        self.inRange = inRange
        self.outRange = outRange
        self.neutralInValue = neutralInValue
        self.outValue = self(neutralInValue)
        
    def __call__(self, x):
        assert self.inRange[0] <= x <= self.inRange[1]
        din = self.inRange[1] - self.inRange[0]
        dout = self.outRange[1] - self.outRange[0]
        outFloat = (x - self.inRange[0]) * dout / din + self.outRange[0]
        return type(self.outRange[0])(outFloat)
    
    def setValue(self, x):
        outValue = self(x)
        self.outValue = outValue
     

class JoystickEmulator(object):
    
    def __init__(self):
        self.vj = vjoy.vj
        self.axes = dict(
            lx=Axis('wAxisX'),
            ly=Axis('wAxisY'),
            rx=Axis('wAxisZ'),
            ry=Axis('wAxisXRot'),
            lt=Axis('wSlider', (0, 1.)),
            rt=Axis('wDial', (0, 1.)),
        )
        def do(action):
            def f(*args, **kwargs):
                out = action(*args, **kwargs)
                self.update()
                return out
            return f
        self.accel = do(self.axes['rt'].setValue)
        self.decel = do(self.axes['lt'].setValue)
        self.yaw = do(self.axes['lx'].setValue)
        self.pitch = do(self.axes['ly'].setValue)
        
    def update(self):
        position = self.vj.generateJoystickPosition(**{
            axis.vjoyName: axis.outValue
            for axis in self.axes.values()
        })
        self.vj.update(position)
        
    def neutral(self):
        for axis in self.axes.values():
            axis.setValue(axis.neutralInValue)
        self.update()


def keydown(combination):
    keyboard.send(combination, do_press=True, do_release=False)


def keyup(combination):
    keyboard.send(combination, do_press=False, do_release=True)


class RateLimiter(object):

    def __init__(self, dt):
        self.dt = dt
        self.tnext = time.time() + dt

    def wait(self):
        t = time.time()
        if t < self.tnext:
            time.sleep(self.tnext - t)


class GTAController(GTA):

    def __init__(self, modelPath):
        GTA.__init__(self)
        import tfUtils
        import tensorflow as tf
        keras = tf.contrib.keras

        self.model = tfUtils.loadModel(modelPath)
        self.model.compile(
            # loss=keras.losses.binary_crossentropy,
            # optimizer=keras.optimizers.Adam(),
            # metrics=['binary_accuracy', 'mean_squared_error', 'mean_absolute_error'],
            loss=keras.losses.mean_absolute_error,
            optimizer=keras.optimizers.Adam(),
            metrics=['binary_accuracy', 'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy'],
        )
        self.goThresh = .19
        self.turnThresh = .19
        self.backThresh = .4
        self.eThresh = .2
        self.escThresh = 1
        self.otherThresh = 1

        self.segmentSize = 24
        self.batch_size = 16

    @property
    def thresholds(self):
        return np.array([
            self.goThresh, self.turnThresh, self.backThresh, self.turnThresh,
            self.eThresh, self.escThresh, self.otherThresh,
        ])

    def predict(self, img=None):
        if img is None:
            img = np.array(self.grab())
        shape = [1]
        shape.extend(img.shape)
        return self.model.predict(img.reshape(tuple(shape)))

    def run(self, stopkey='esc', dt=.2):

        self.SetAsForegroundWindow()

        rateLimiter = RateLimiter(dt)

        try:
            self.keyRecorder.stop()
        except ValueError:
            pass
        trap = BoolKeyTrap(key=stopkey)
        print('Processing images ...')
        self.nprint = 0
        while True:
            if trap.trapped:
                print('Trapped. Exiting run loop.')
                break
            self.setControls(self.predict())
            rateLimiter.wait()

    def setControls(self, state):
        state = state > self.thresholds
        for s, key in zip(state.ravel(), self.keyNames):
            if s:
                print(key, end='')
                if hasattr(self, 'nprint') and self.nprint > 32:
                    print('')
                    print(time.time(), end=' ')
                keydown(key)
            else:
                keyup(key)

    def continuouslyLearn(self, tqdm_notebook=False, trapKey='end', **kwargs):
        keyboard.remove_all_hotkeys()
        keyboard.clear_all_hotkeys()
        trap = BoolKeyTrap(key=trapKey)
        self.startRecordingKeyboard()
        while True:
            if trap.trapped:
                break
            self.clearHistory()
            # print('Grabbing %d images ...' % self.segmentSize, end=' ')
            self.grabImagesUntilInterrupted(trapKey=None, limit=self.segmentSize, tqdm_notebook=None, **kwargs)
            # print('done. Got %d.' % len(self.imgs))
            self.learnFromImageset(self.imgs, self.states, tqdm_notebook=tqdm_notebook)
        self.stopRecordingKeyboard()

    def learnFromImageset(self, images, states, tqdm_notebook=False):
        if isinstance(images, list):
            images = np.stack(images)
        if isinstance(states, list):
            states = np.stack(states)

        nsegments = np.math.ceil(float(len(images)) / self.segmentSize)
        bar = getBar(tqdm_notebook)
        for segment in bar(range(nsegments)):
            i1 = segment*self.segmentSize
            i2 = min((segment+1)*self.segmentSize, len(images))
            self.model.fit(
                np.array(images[i1:i2]), np.array(states[i1:i2]),
                epochs=1,
                batch_size=self.batch_size,
                verbose=0,
            )

    def save(self, modelPath):
        import tfUtils
        tfUtils.saveModel(self.model, modelPath)

# import xinput
# s = xinput.sample_first_joystick()
    # gives output like this: -> 
    # axis l_thumb_x 0.4524605172808423
    # axis l_thumb_y -0.08278019378957809
    # axis l_thumb_x 0.3930418860151064
    # axis l_thumb_x 0.33191424429694055
    # axis right_trigger 0.10196078431372549
    # axis right_trigger 0.23137254901960785
    # axis right_trigger 0.5019607843137255

# joysticks = xinput.XInputJoystick.enumerate_devices()
# device_numbers = list(map(xinput.attrgetter('device_number'), joysticks))
# j = joysticks[0]

#import inputs
#pad0 = inputs.devices.gamepads[0]
