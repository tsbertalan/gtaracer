from gta.gameInputs.gamepad import JoystickEmulator

class ControlInputs(object):

    def __init__(self):

        self.gamepadEmulator = JoystickEmulator()
        self.oldAxes = [0] * 6

    def applyControlState(self, controlVector):
        axes = controlVector[:6]
        keys = 'lx', 'ly', 'lt', 'rx', 'ry', 'rt'
        for old, new, key in zip(self.oldAxes, axes, keys):
            if old != new:
                self.gamepadEmulator.axes[key].setValue(new)
        self.gamepadEmulator.update()
        self.oldAxes = axes

        lx, ly, lt, rx, ry, rt = axes
