import numpy as np
from simple_pid import PID
from tqdm import tqdm
import time

from sys import path
from os.path import join, expanduser
path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))

import gta.recording.vision
import gta.gameInputs.gamepad

from PIL import Image
import cv2


class Controller:

    def __init__(self):
        self.gameWindow = gta.recording.vision.GtaWindow()

        # grab = np.array(gta.recording.vision.win32_grab2(self.gameWindow._hwnd, self.gameWindow.getBbox()))
        # print(grab.dtype)
        # cv2.imshow(
        #     'win32_grab',
        #     grab
        # )
        # cv2.waitKey(1)

        self.angle_weight = 4.2
        self.offset_weight = .5
        self.pid = PID(0.15, 0.03, 0.1, setpoint=0)
        self.pid.proportional_on_measurement = False
        self.pid.output_limits = (-.45, .45)
        self.gpad = gta.gameInputs.gamepad.Gamepad()
        self.throttle = 0.3
        self.brake = .0#5
        self.last_cycle_time = time.time()
        self._n_too_few = 0
        self.fit_order = 1

    def get_cte(self):
        tm = self.gameWindow.track_mask
        rows, cols = tm.shape[:2]
        points = np.argwhere(tm)
        if len(points) < 10:
            self._n_too_few += 1
            if self._n_too_few > 20:
                raise StopIteration
            else:
                raise ValueError
        elif len(points) > 2000:
            raise ValueError
        coeffs = np.polyfit(*points.T, deg=self.fit_order)
        poly = np.poly1d(coeffs)
        slope = poly.deriv()

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.imshow(tm)
        # xl = ax.get_xlim(); yl = ax.get_ylim()
        # y = np.linspace(0, rows, 32)
        # x = poly(y)
        # ax.plot(x, y, color='green')
        # ax.scatter(self.gameWindow.car_origin[1], self.gameWindow.car_origin[0], marker='+', s=500)
        # plt.show()

        tmu = tm.astype('uint8')*255
        # tmu = np.array(self.gameWindow.micromap)
        scale = 2.0
        cv2.imshow('track_mask', 
            # tmu,
            cv2.resize(tmu, None, fx=scale, fy=scale)
        )
        cv2.waitKey(1)  

        offset = self.gameWindow.car_origin[1] - poly(self.gameWindow.car_origin[0])
        angle = np.arctan(slope(self.gameWindow.car_origin[0]))
        ow = offset * self.offset_weight
        aw = angle  * self.angle_weight
        sign = lambda x: x >= 0
        if sign(ow) == sign(aw):
            print('AGREE', end=' ')
        elif abs(ow) > abs(aw):
            print('OFFSET', end=' ')
        else:
            print('ANGLE', end=' ')
        if ow + aw > 0:
            print('LEFT', end=' ')
        else:
            print('RIGHT', end=' ')
        # print('offset=%.2f -> %.2f' % (offset, ow))
        # print('angle=%.2f -> %.2f' % (angle, aw))
        return ow + aw

    def compute_control(self):
        try:
            err = self.get_cte()
            ct = self.pid(err)
        except ValueError:
            ct = self.pid._last_output
        return 0 if ct is None else ct

    def step(self):
        steer = self.compute_control()
        pom = '(PoM)' if self.pid.proportional_on_measurement else ''
        p,i,d = self.pid.components
        print('pid: %.2f%s,%.2f,%.2f' % (p, pom, i, d), end=' ')
        print('steer: %.2f' % steer, end=' ')
        self.gpad(steer=steer, accel=self.throttle, decel=self.brake)
        t = time.time()
        print('dt=%.2f' % (t-self.last_cycle_time))
        self.last_cycle_time = t
        return(steer)

    def stop(self):
        print('Stopping')
        self.gpad(steer=0, accel=0, decel=0)

    def __del__(self):
        self.stop()


if __name__ == '__main__':
    from pprint import pprint
    controller = Controller()

    # controller.gameWindow.micromap.show()

    # Image.fromarray(controller.gameWindow.track_mask).show()

    while True:
        controller.step()
    
