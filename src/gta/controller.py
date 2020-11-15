#!/usr/bin/env python

import numpy as np
from simple_pid import PID
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logger = logging.Logger('controller')
logger.setLevel(logging.INFO)

from sys import path
from os.path import join, expanduser
path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))

import gta.recording.vision
import gta.gameInputs.gamepad

from PIL import Image
import cv2

DEFAULT_CAR_THROTTLE = 0.4

class PointsError(ValueError):
    pass


class Controller:

    def __init__(self, throttle=None, minimum_throttle=None, steer_limit=0.5, kp=.1, ki=0.02, kd=0.05, 
        pretuned=None, error_kind='cte', draw_on_basemap=False):
        self.error_kind = error_kind
        print('Using', error_kind, 'error.')
        steer_limit = abs(steer_limit)
        self.gameWindow = gta.recording.vision.GtaWindow()
        self.draw_on_basemap = draw_on_basemap
        self._too_few_points_maxiter = 500

       
        self.angle_weight = 4
        self.offset_weight = .5
        if pretuned is not None:
            print('Using', pretuned, 'PID pretuning.')
            kp, ki, kd = dict(
                car=(.22, .03, .1), 
                slowcarrace=(1.5, .1, .01), 
                race=(.8, .1, .01), 
                rcrace=(.8, .05, .01), 
                boatrace=(1.6, .06, .005),
                boat=(.8, 0.02, 0), 
                sailboat=(.9, .05, 0)
            )[pretuned]
            self.filters_present = dict(
                car=['all'],
                race=['all'],
                slowcarrace=['all'],
                rcrace=['all'],
                boatrace=['all'],
                boat=['purple_cross'],
                sailboat=['purple_cross'],
            )[pretuned]
        else:
            self.filters_present = ['all']
            
        self.pid = PID(kp, ki, kd, setpoint=0)
        self.pid.proportional_on_measurement = False
        self.steer_limit = steer_limit
        self.pid.output_limits = (-steer_limit, steer_limit)
        self.gpad = gta.gameInputs.gamepad.Gamepad()

        self.throttle_nominal = DEFAULT_CAR_THROTTLE if throttle is None else throttle
        if minimum_throttle is None:
            if pretuned == 'sailboat':
                minimum_throttle = .6
            else:
                minimum_throttle = .4
        self.minimum_throttle = min(self.throttle_nominal, minimum_throttle)
        self.throttle_fuzz = 0.01

        self.brake = .0
        self.last_cycle_time = time.time()
        self._n_too_few = 0
        self.fit_order = 3
        self.slope_vertical_offset = -6
        self.cte_history = []
        self.control_history = []
        self._last_good_error = 0
        self._last_steering = 0

    def get_error(self):
        if self.error_kind == 'cte':
            out = self.get_cte()
        else:
            out = self.get_heading_error()
        if np.isnan(out):
            return self._last_good_error
        else:
            self._last_good_error = out
            return out

    def get_heading_error(self):
        tm, basemap = self.gameWindow.get_track_mask(basemap_kind='minimap', do_erode=False, filters_present=self.filters_present)
        tm = np.array(tm)

        rows, cols = tm.shape[:2]
        points = np.argwhere(tm)
        scale = 2.0
        if self.draw_on_basemap:
            display = np.array(self.gameWindow.minimap)
        else:
            display = tm
            display = np.copy(display if 'uint' in str(display.dtype) else display.astype('uint8')*255)
            display = np.dstack([display, display, display])

        #display = display + self.gameWindow.minimap

        try:
            x0, y0 = self.gameWindow.car_origin_minimap

            # Do some simple outlier rejection.
            xm, ym = np.mean(points, axis=0)
            distance_from_mean = np.linalg.norm(points - np.array([[xm, ym]]), axis=1)
            distance_threshold = self.gameWindow.wscale(14., 0)[0]
            near_mean = distance_from_mean < distance_threshold

            x1, y1 = np.mean(points[near_mean], axis=0)

            target_behind_us = x1 > x0

            if x0 > x1:
                x0,y0,x1,y1 = x1,y1,x0,y0

            dx = (x1 - x0)
            dy = (y1 - y0)

            slope = dy / dx

            self.squared_distance_to_waypoint = np.sqrt(dx ** 2 + dy ** 2)

            self.dot(display, [x0, y0])
            self.dot(display, [x1, y1], color=(255, 0, 0))
            xl = np.arange(int(x0), int(x1))
            yl = ((xl-x0) * slope + y0).astype('uint16')
            ok = np.logical_and(yl > 0, yl < display.shape[1])
            display[xl[ok], yl[ok], :] = 255

        except ValueError:
            pass

        cv2.imshow('track_mask', 
            cv2.resize(cv2.cvtColor(display, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale)
        )
        cv2.waitKey(1)

        if points.size < 1:
            self._n_too_few += 1
            if self._n_too_few > self._too_few_points_maxiter:
                raise StopIteration
            else:
                raise PointsError('Too few points!')
        else:
            if target_behind_us:
                return np.arctan(np.inf)
            else:
                return np.arctan(slope)

    @staticmethod
    def dot(display, location, radii=(1, 1), color=(0, 0, 255)):
        cr, cc = np.asarray(location).astype('int')
        for r in range(cr-radii[0], cr+radii[0]+1):
            for i, v in enumerate(color):
                display[int(r), np.arange(int(cc-radii[1]), int(cc+radii[1]+1)), i] = v

    def get_cte(self):
        tm, basemap = self.gameWindow.track_mask
        tm = np.array(tm)
        rows, cols = tm.shape[:2]
        points = np.argwhere(tm)
        if len(points) < 10:
            self._n_too_few += 1
            if self._n_too_few > self._too_few_points_maxiter:
                raise StopIteration
            else:
                raise PointsError('Too few points!')
        elif len(points) > 2000:
            raise PointsError('Too many points!')
        coeffs = np.polyfit(*points.T, deg=self.fit_order)
        poly = np.poly1d(coeffs)
        slope = poly.deriv()

        if self.draw_on_basemap:
            display = basemap
        else:
            display = tm.astype('uint8')*255
            rows, cols = display.squeeze().shape
            display = np.dstack([display, display, display])
            
        Y = np.arange(self.gameWindow.car_origin[0]).astype(int)
        X = poly(Y).astype(int)
        ok = np.logical_and(X>0, X<cols)
        display[Y[ok], X[ok], 0] = 0
        display[Y[ok], X[ok], 1] = 255
        display[Y[ok], X[ok], 2] = 0
        
        self.dot(display, self.gameWindow.car_origin)
        self.dot(display,
            [
                self.gameWindow.car_origin[0]+self.slope_vertical_offset,
                self.gameWindow.car_origin[1],
            ],
            color=(0, 255, 255),
            radii=(1, 3),
        )
        scale = 2.0
        cv2.imshow('track_mask', 
            # display,
            cv2.resize(display, None, fx=scale, fy=scale)
        )
        cv2.waitKey(1)  

        offset = self.gameWindow.car_origin[1] - poly(self.gameWindow.car_origin[0])
        angle = np.arctan(slope(self.gameWindow.car_origin[0]+self.slope_vertical_offset))
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
        self.cte_history.append(ow+aw)
        return ow + aw

    def compute_control(self):
        try:
            err = self.get_error()
            ct = self.pid(err)
        except PointsError as e:
            logger.info(str(e))
            ct = self.pid._last_output
        return 0 if ct is None else ct

    def compute_throttle(self):
        x1 = self.steer_limit
        y1 = self.minimum_throttle

        x2 = 0
        y2 = self.throttle_nominal

        slope = (y2 - y1) / (x2 - x1)

        x = abs(self._last_steering)
        y = (x - x2) * slope + y2

        # if hasattr(self, 'squared_distance_to_waypoint'):
        #     print(self.squared_distance_to_waypoint)

        if self.throttle_fuzz == 0:
            return y
        else:
            return float(np.random.normal(loc=y, scale=self.throttle_fuzz))

    def step(self):
        self._last_steering = steer = self.compute_control()
        throttle = self.compute_throttle()
        print('thr=%.2f' % throttle, end=',')
        applied = self.gpad(steer=steer, accel=throttle, decel=self.brake)
        steer = applied[0]

        pom = '(PoM)' if self.pid.proportional_on_measurement else ''
        p,i,d = self.pid.components
        self.control_history.append([p, i, d, steer])
        print('pid: %.2f%s,%.2f,%.2f' % (p, pom, i, d), end=' ')
        print('steer: %.2f' % steer, end=' ')
        t = time.time()
        print('dt=%.2f' % (t-self.last_cycle_time))
        self.last_cycle_time = t
        
        return applied

    def stop(self):
        self.gpad(steer=0, accel=0, decel=0)

    def __del__(self):
        self.stop()

    def plot(self, save=False):
        fig, ax = plt.subplots()
        ax.plot([e/10. for e in self.cte_history], label=r'$\epsilon/10$')
        for k, label in enumerate(['$P$', '$I$', '$D$', r'$\Sigma$']):
            ax.plot([pids[k] for pids in self.control_history], label=label)
        ax.legend(loc='best')
        ax.set_xlabel('Step')
        ax.set_title('$k_p={kp}$, $k_i={ki}$, $k_d={kd}$'.format(
            kp=self.pid.Kp,
            ki=self.pid.Ki,
            kd=self.pid.Kd,
        )
        +
        ('' if not self.pid.proportional_on_measurement else ' (PoM)')
        )

        a, b = ax.get_ylim()
        ax.set_ylim(max(-1, a), min(1, b))

        if save:
            for ext in 'png', 'pdf':
                fig.savefig('kp{kp}_ki_{ki}_kd{kd}_{time}.{ext}'.format(
                    kp=self.pid.Kp,
                    ki=self.pid.Ki,
                    kd=self.pid.Kd,
                    ext=ext,
                    time=time.time(),
                ))


def main():

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--throttle', type=float, default=None)
    parser.add_argument('--minimum_throttle', type=float, default=None)
    parser.add_argument('--show_plot', action='store_true', default=False)
    parser.add_argument('--save_plot', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='car')
    parser.add_argument('--error_kind', type=str, default=None)
    parser.add_argument('--steer_limit', type=float, default=None)
    # parser.add_argument('--kp', type=float, default=0.10)
    # parser.add_argument('--ki', type=float, default=0.02)
    # parser.add_argument('--kd', type=float, default=0.05)
    parser.add_argument('--pid_tuning', type=str, default=None)
    args = parser.parse_args()

    # For debugging:
    # args.mode = 'boatrace'

    car = args.mode.lower() == 'car'
    race = args.mode.lower() == 'race'
    boatrace = args.mode.lower() == 'boatrace'
    if args.throttle is None: args.throttle = DEFAULT_CAR_THROTTLE if (car or race) else 0.6 if boatrace else 1.0
    if args.error_kind is None: args.error_kind = 'cte' if car else 'heading'
    if args.steer_limit is None: args.steer_limit = .5 if (car or race) else 1.0
    if args.pid_tuning is None: args.pid_tuning = args.mode.lower()

    controller = Controller(
        throttle=args.throttle, error_kind=args.error_kind, 
        # kp=args.kp, ki=args.ki, kd=args.kd,
        pretuned=args.pid_tuning,
        steer_limit=args.steer_limit,
        minimum_throttle=args.minimum_throttle
    )

    while True:
        try:
            controller.step()
            
        except BaseException as e:

            if isinstance(e, KeyboardInterrupt):
                logger.info('User interrupted.')
                break

            else:
                logger.warning('Stopping:', e)
                if args.show_plot or args.save_plot:
                    controller.stop()
                    controller.plot(save=args.save_plot)
                    if args.show_plot:
                        plt.show()
                break
   

if __name__ == '__main__':
    main()
