import numpy as np
from sys import path
from os.path import join, expanduser
path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))

import vjoy
from time import sleep

def map_axis(x, in_range, out_range):
    y = (x - in_range[0]) / (in_range[1] - in_range[0])
    z = y * (out_range[1] - out_range[0]) + out_range[0]
    # print(x, 'in', in_range, '->', z, 'in', out_range)
    return z

class Axis:

    def __init__(self, name, low=0, high=65534./2, nominal_range=(-1, 1), neutral=None, purpose=None, 
        common_axis_name=None, kind=int, input_clamp=(-np.inf, np.inf), deadzone=0):
        self.name = name
        self.low = float(low)
        self.high = float(high)
        if neutral is None:
            neutral = (self.low + self.high) / 2.
        self.nominal_range = tuple([float(x) for x in nominal_range])
        self.neutral = neutral
        self.neutral_nominal = self.native2nominal(neutral)
        self.purpose = purpose
        self.common_axis_name = common_axis_name
        self.kind = kind
        self.last_input = self.native2nominal(self.neutral)
        self.input_clamp = tuple([float(x) for x in input_clamp])
        self.deadzone = deadzone

    def nominal2native(self, input_nominal):
        out = self.kind(map_axis(input_nominal, self.nominal_range, (self.low, self.high)))
        # print(self.purpose, ': Mapping nominal', input_nominal, 'to native', out)
        return out

    def native2nominal(self, input_native):
        return map_axis(input_native, (self.low, self.high), self.nominal_range)

    def __call__(self, input_nominal):
        input_nominal = min(self.input_clamp[1], max(self.input_clamp[0], input_nominal))
        self.last_input = input_nominal
        value = self.nominal2native(input_nominal)
        return {self.name: value}

    def deadzone_filter(self, input_nominal):
        x = input_nominal
        z = self.neutral_nominal
        d = self.deadzone
        if d == 0 or x == z:
            return x
        else:
            # Two thresholds:
            b, a = z + d, z - d

            if x > b or x < a:
                # The input is not objectionable.
                return x

            else:
                # Saturate at one of the thresholds:
                # choose the one closest to the input.
                if abs(a - x) < abs(b - x):
                    return a
                else:
                    return b

class Button:

    def __init__(self, index, name, vj):
        self.index = index
        self.name = name
        self.vj = vj
        self.value = 0

    def tap(self, duration=.2):
        self.set(1)
        sleep(duration)
        self.set(0)

    def set(self, value):
        self.value = int(bool(value))
        self.vj.setButton(self.index, self.value)

class Gamepad:

    def __init__(self):
        self.vj = vjoy.vJoy()
        self.vj.open()
        self.steering_dead_zone = 0.0

        # X360ce should have
        #     Left Trigger set to "Axis 2",
        #     Right Trigger set to "Axis 3", and
        #     Left Stick X set to "Axis 1".

        using_x360ce = True
        if using_x360ce:
            axis_codes = 'wAxisX', 'wAxisY', 'wAxisZ'
        
        self._axes = [
            Axis(axis_codes[0], neutral=None, nominal_range=(-1, 1), purpose='steer', common_axis_name='Left Stick X',  input_clamp=[-1, 1], deadzone=self.steering_dead_zone),
            Axis(axis_codes[1], neutral=6000, nominal_range=(0, 1),  purpose='decel', common_axis_name='Left Trigger',  input_clamp=[0, 1]),
            Axis(axis_codes[2], neutral=6000, nominal_range=(0, 1),  purpose='accel', common_axis_name='Right Trigger', input_clamp=[0, 1]),
        ]
        for ax in self._axes:
            setattr(self, ax.purpose, ax)

        self._buttons = [
            Button(1, 'a', self.vj),
            Button(2, 'b', self.vj),
            Button(3, 'x', self.vj),
            Button(4, 'y', self.vj),
            Button(5, 'lb', self.vj),
            Button(6, 'rb', self.vj),
            Button(7, 'start', self.vj),
        ]
        for button in self._buttons:
            setattr(self, button.name, button)

    def _update(self, xstick_lt_rt):
        positions = {}
        for val, axis in zip(xstick_lt_rt, self._axes):
            positions.update(axis(val))
        # print(positions)
        joypos = self.vj.generateJoystickPosition(**positions)
        self.vj.update(joypos)
        return xstick_lt_rt

    # def _get_axis_by_purpose(self, purpose):
    #     return dict(
    #         steer=self._axes[0],
    #         decel=self._axes[1],
    #         accel=self._axes[2],
    #     ).get(purpose, None)

    #     # for ax in self._axes:
    #     #     if ax.purpose == purpose:
    #     #         return ax
        
    def __call__(self, steer=None, decel=None, accel=None):
        values = []
        for ax, val in zip(self._axes, [steer, decel, accel]):
            values.append(
                ax.deadzone_filter(val) if val is not None
                else ax.last_input
            )
        self._update(values)
        return values


if __name__ == '__main__':
    gp = Gamepad()

    # sleep(4)

    button_index = 2

    # gp.vj.setButton(button_index, 1)
    # gp.a = 1


    for _ in range(100):
        # gp(accel=0, decel=0, steer=0)
        # gp(steer=1, decel=0, accel=1)
        # gp()
        sleep(.01)

    # gp(0, 0, 0)

    # gp.vj.setButton(button_index, 0)
    # gp.a = 0

    gp.b.tap(.5)

    # for _ in range(10):
    #     gp(accel=.5, decel=-)
    #     sleep(.01)

