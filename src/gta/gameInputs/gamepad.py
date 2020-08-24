from sys import path
from os.path import join, expanduser
import numpy as np
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
        common_axis_name=None, kind=int, input_clamp=(-np.inf, np.inf)):
        self.name = name
        self.low = float(low)
        self.high = float(high)
        if neutral is None:
            neutral = (self.low + self.high) / 2.
        self.nominal_range = tuple([float(x) for x in nominal_range])
        self.neutral = neutral
        self.purpose = purpose
        self.common_axis_name = common_axis_name
        self.kind = kind
        self.last_input = self.native2nominal(self.neutral)
        self.input_clamp = tuple([float(x) for x in input_clamp])

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


class Gamepad:

    def __init__(self):
        self.vj = vjoy.vJoy()
        self.vj.open()

        # X360ce should have
        #     Left Trigger set to "Axis 2",
        #     Right Trigger set to "Axis 3", and
        #     Left Stick X set to "Axis 1".

        self._axes = [
            Axis('wAxisX', neutral=None, nominal_range=(-1, 1), purpose='steer', common_axis_name='Left Stick X',  input_clamp=[-1, 1]),
            Axis('wAxisY', neutral=6000, nominal_range=(0, 1),  purpose='decel', common_axis_name='Left Trigger',  input_clamp=[0, 1]),
            Axis('wAxisZ', neutral=6000, nominal_range=(0, 1),  purpose='accel', common_axis_name='Right Trigger', input_clamp=[0, 1]),
        ]
        sleep(.2)

    def _update(self, xstick_lt_rt):
        positions = {}
        for val, axis in zip(xstick_lt_rt, self._axes):
            positions.update(axis(val))
        print(positions)
        joypos = self.vj.generateJoystickPosition(**positions)
        self.vj.update(joypos)

    def _get_axis_by_purpose(self, purpose):
        return dict(
            steer=self._axes[0],
            decel=self._axes[1],
            accel=self._axes[2],
        ).get(purpose, None)

        # for ax in self._axes:
        #     if ax.purpose == purpose:
        #         return ax
        
    def __call__(self, steer=None, decel=None, accel=None):
        return self._update([
            steer if steer is not None else self._get_axis_by_purpose('steer').last_input,
            decel if decel is not None else self._get_axis_by_purpose('decel').last_input,
            accel if accel is not None else self._get_axis_by_purpose('accel').last_input,
        ])


if __name__ == '__main__':
    gp = Gamepad()

    # sleep(4)

    for _ in range(100):
        # gp(accel=0, decel=0, steer=0)
        gp(steer=1, decel=0, accel=0)
        # gp()
        sleep(.01)

    # for _ in range(10):
    #     gp(accel=.5, decel=-)
    #     sleep(.01)

