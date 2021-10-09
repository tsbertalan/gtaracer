#!/usr/bin/env python

import numpy as np
from simple_pid import PID
import time

import matplotlib.pyplot as plt

import logging

logformat = '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
datefmt = '%H:%M:%S'
logging.basicConfig(format=logformat, datefmt=datefmt, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from sys import path
from os.path import join, expanduser
path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))

import gta.recording.vision
import gta.gameInputs.gamepad
from gta.random_destinations import change_gps

from PIL import Image
import cv2


class PointsError(ValueError):
    pass


class IOController:

    def __init__(self, throttle=None, minimum_throttle=None, steer_limit=0.5, kp=.1, ki=0.02, kd=0.05, 
        pretuned='car', filters_present='carmission', error_kind='cte', draw_on_basemap=False,
        do_perspective_transform=True, dry_run=False, use_minimap=False,
        throttle_disabled=False,
        ):
        self.error_kind = error_kind
        if error_kind == 'cte':
            self._too_few_points_minpoints = 1
        else:
            self._too_few_points_minpoints = 10
        logger.info('Error kind: {}'.format(error_kind))
        steer_limit = abs(steer_limit)
        self.gameWindow = gta.recording.vision.GtaWindow()
        self.draw_on_basemap = draw_on_basemap
        self._too_few_points_maxiter = TOO_FEW_POINTS_MAXITER_DEFAULT
        self.n_randomizations = 0
        self.last_randomization = time.time()
        self.min_randomize_interval = 10.
        self._max_randomizations = 5

        self.do_perspective_transform = do_perspective_transform

        self.dry_run = dry_run
        self.use_minimap = use_minimap

        self.filter_display_level = 128
        self.tentacle_color = 128, 255, 0  # r, g, b
       
        self.angle_weight = 4
        self.offset_weight = .5
        assert pretuned is not None

        logger.info('PID pretuning: {}'.format(pretuned))
        kp, ki, kd = dict(
            car=(.22, .03, .1), 
            chase=(.18, .005, .35), 
            heavy=(.3, .03, .1),
            mission=(.22, .03, .01), 
            carmission=(.22, .03, .01),
            slowcarrace=(1.5, .1, .01), 
            race=(.8, .1, .01), 
            rcrace=(.8, .05, .01), 
            boatrace=(1.6, .06, .005),
            boat=(1.6, 0.06, 0.005), 
            sailboat=(.9, .05, 0)
        )[pretuned]

        if self.do_perspective_transform:
            # Compensate for changed units after perspective transform.
            perspective_factor = 2.0
            kp *= perspective_factor
            ki *= perspective_factor
            kd *= perspective_factor

        if filters_present is not None:
            filter_options = dict(
                car=['magenta_line'],
                mission=['yellow_line', 'green_line', 'sky_line'],
                carmission=['magenta_line', 'yellow_line', 'green_line', 'sky_line'],
                race=['all'],
                slowcarrace=['all'],
                boat=['all'],
                rcrace=['all'],
                boatrace=['all'],
                walk=['purple_cross'],
                sailboat=['purple_cross', 'race_dots'],
            )
            self.filters_present = set()
            if 'car' in filters_present:
                self.filters_present.update(filter_options['car'])
            if 'mission' in filters_present:
                self.filters_present.update(filter_options['mission'])
            if 'race' in filters_present or 'slowcarrace' in filters_present or 'rcrace' in filters_present:
                self.filters_present.add('all')
            if 'boat' in filters_present:
                self.filters_present.update(filter_options['boat'])
            if 'race' in filters_present:
                self.filters_present.update(filter_options['race'])
                
        if len(self.filters_present) == 0:
            self.filters_present = ['all']
        else:
            self.filters_present = list(self.filters_present)

        logger.info('filters_present: {}'.format(self.filters_present))

        print('Logger level=', logging.getLevelName(logger.getEffectiveLevel()))
        
        logger.info('PID: kp={}, ki={}, kd={}'.format(kp, ki, kd))
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
        self.throttle_disabled = throttle_disabled

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
        tm, basemap = self.gameWindow.get_track_mask(
            basemap_kind='minimap', do_erode=False, filters_present=self.filters_present,
            do_perspective_transform=self.do_perspective_transform
        )
        tm = np.array(tm)

        rows, cols = tm.shape[:2]
        points = np.argwhere(tm)
        scale = 2.0
        if self.draw_on_basemap:
            display = np.array(self.gameWindow.minimap)
            # Convert CV2 to Matplotlib colors
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        else:
            display = tm
            display = np.copy(display if 'uint' in str(display.dtype) else display.astype('uint8')*self.filter_display_level)
            display = np.dstack([display, display, display])

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

            self.draw_dot(display, [x0, y0])
            self.draw_dot(display, [x1, y1], color=(self.filter_display_level, 0, 0))
            xl = np.arange(int(x0), int(x1))
            yl = ((xl-x0) * slope + y0).astype('uint16')
            ok = np.logical_and(yl > 0, yl < display.shape[1])
            display[xl[ok], yl[ok], :] = self.tentacle_color

        except ValueError:
            pass

        cv2.imshow('track_mask', 
            cv2.resize(cv2.cvtColor(display, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale)
        )
        cv2.waitKey(1)

        self.last_point_count = points.size
        if points.size < 1:
            self._n_too_few += 1
            if self._n_too_few > self._too_few_points_maxiter:
                raise StopIteration('Saw too few points too many times; not continuing controller.')
            else:
                raise PointsError('Too few points (try again later).')
        else:
            if target_behind_us:
                return np.arctan(np.inf)
            else:
                return np.arctan(slope)
    
    @property
    def has_few_points(self):
        if not hasattr(self, 'last_point_count'):
            return True
        else:
            if self.last_point_count < self._too_few_points_minpoints:
                return True
            else:
                return False

    def randomize_gps(self):
        if time.time() - self.last_randomization > self.min_randomize_interval:
            logger.info('Randomizing GPS destination')
            self.last_randomization = time.time()
            self.n_randomizations += 1
            change_gps(self.gpad)

    @staticmethod
    def draw_dot(display, location, radii=(1, 1), color=(0, 0, 255)):
        cr, cc = np.asarray(location).astype('int')
        for r in range(cr-radii[0], cr+radii[0]+1):
            for i, v in enumerate(color):
                display[int(r), np.arange(int(cc-radii[1]), int(cc+radii[1]+1)), i] = v

    def get_cte(self):
        minimap = self.use_minimap  # Whether to keep the larger minimap or zoom in to the smaller "micromap".
        origin = (
            self.gameWindow.car_origin_micromap_perspectiveTransformed
            if self.do_perspective_transform else
            self.gameWindow.car_origin
        ) if not minimap else (
            self.gameWindow.car_origin_minimap_perspectivetransformed 
            if self.do_perspective_transform else
            self.gameWindow.car_origin_minimap
        )
        tm, basemap = self.gameWindow.get_track_mask(
            basemap_kind='minimap' if minimap else 'micromap', do_erode=False, 
            filters_present=self.filters_present,
            do_perspective_transform=self.do_perspective_transform,
        )
        tm = np.array(tm)
        rows, cols = tm.shape[:2]
        points = np.argwhere(tm)
        self.last_point_count = len(points)
        if self.has_few_points:
            self._n_too_few += 1
            if self._n_too_few > self._too_few_points_maxiter:
                raise StopIteration('Saw too few points too many times; not continuing controller.')
            else:
                raise PointsError('Too few points!')
        elif len(points) > 2000:
            raise PointsError('Too many points!')

        else:
            coeffs = np.polyfit(*points.T, deg=self.fit_order)
            poly = np.poly1d(coeffs)
            slope = poly.deriv()

        if self.draw_on_basemap:
            display = basemap
            # Convert CV2 to Matplotlib colors
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        else:
            display = tm.astype('uint8')*self.filter_display_level
            rows, cols = display.squeeze().shape
            display = np.dstack([display, display, display])

        if 'poly' in locals():

            # Take y values from the origin up to the highest point.
            Y = np.arange(origin[0]).astype(int)

            # Evaluate the corresponding x values.
            X = poly(Y).astype(int)
            ok = np.logical_and(X>0, X<cols)
            display[Y[ok], X[ok], 0] = self.tentacle_color[2]
            display[Y[ok], X[ok], 1] = self.tentacle_color[1]
            display[Y[ok], X[ok], 2] = self.tentacle_color[0]

            # Use a morphological filter to thicken the line.
            kernel = np.ones((3, 3), np.uint8)
            display = cv2.dilate(display, kernel, iterations=1)
        
        self.draw_dot(display, origin)
        self.draw_dot(display,
            [
                origin[0]+self.slope_vertical_offset,
                origin[1],
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

        if 'slope' in locals():
            offset = origin[1] - poly(origin[0])
            angle = np.arctan(slope(origin[0]+self.slope_vertical_offset))
            ow = offset * self.offset_weight
            aw = angle  * self.angle_weight
            err = ow + aw
        else:
            err = 0

        self.cte_history.append(err)
        return err

    def compute_control(self):
        try:
            err = self.get_error()
            ct = self.pid(err)
        except PointsError as e:
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

        if self.throttle_fuzz == 0:
            return y
        else:
            return float(np.random.normal(loc=y, scale=self.throttle_fuzz))

    def step(self):
       
        self._last_steering = steer = self.compute_control()

        if self.throttle_disabled:
            throttle = None
        else:
            throttle = self.compute_throttle()
        if not self.dry_run:
            applied = self.gpad(steer=steer, accel=throttle, decel=self.brake)
            steer = applied[0]

            pom = '(PoM)' if self.pid.proportional_on_measurement else ''
            p,i,d = self.pid.components
            self.control_history.append([p, i, d, steer])
            t = time.time()
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


class RangeHeadingController(IOController):
    pass


def make_three_channels(mono, as_Image=True):
    if isinstance(mono, Image.Image):
        mono = np.array(mono)
    out = np.dstack([mono, mono, mono])
    if as_Image:
        out = Image.fromarray(out)
    return out


def bool2uint8(arr):
    return arr.astype('bool').astype('uint8') * 255


class OptimalController(IOController):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)

    def get_cost_map(self, smaller_edge_goal=100, drivable_reward_level=.75, objective_reward_level=1.0):
        tm, basemap = self.gameWindow.get_track_mask(basemap_kind='micromap', do_erode=False, filters_present=self.filters_present)

        # Scale the image to some small size.
        smaller_edge_goal = 100
        smallsize = min(basemap.shape[:2])
        scale = float(smaller_edge_goal) / smallsize
        basemap_small = cv2.resize(basemap, None, fx=scale, fy=scale)
        tm_small = cv2.resize(bool2uint8(tm), None, fx=scale, fy=scale)

        # Take the value channel.
        base_hsv = cv2.cvtColor(basemap_small, cv2.COLOR_RGB2HSV)
        value = base_hsv[:, :, 2]

        # Do some adaptive thresholding.
        from functools import reduce
        drivable = reduce(np.logical_and, [
            # base_hsv[:, :, 1] < 100,  # Also only grey parts
            # value > 100,
            cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            value > value.mean(), 
            cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 165, 8)
            #cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 165, 8)
        ])

        # Waypoints
        objectives = cv2.dilate(tm_small.astype('float32'), np.ones([5,5]), iterations=1).astype('bool')

        # Mix
        costmap = (
            # cv2.GaussianBlur(drivable.astype('float32'), (51, 51), cv2.BORDER_CONSTANT) * drivable_reward_level
            drivable.astype('float32') * drivable_reward_level
            +
            objectives.astype('float32') * objective_reward_level
        )

        # Blur
        blurred_costmap = cv2.GaussianBlur(costmap, (31, 31), cv2.BORDER_REFLECT)

        cv2.imshow('track_mask', blurred_costmap)
        cv2.waitKey(1)

        return blurred_costmap

    def compute_control(self):
        cost_map = self.get_cost_map()
        
        display = cost_map
        display = np.copy(display if 'uint' in str(display.dtype) else display.astype('uint8')*255)
        display = make_three_channels(display, as_Image=False)
        # scale = 2.0
        # cv2.imshow('track_mask', 
        #     cv2.resize(cv2.cvtColor(display, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale)
        # )
        # cv2.waitKey(1)
        return 0
        
    def step(self):
        self._last_steering = steer = self.compute_control()
        throttle = self.compute_throttle()
        logger.debug('thr=%.2f' % throttle)
        applied = self.gpad(steer=steer, accel=throttle, decel=self.brake)
        steer = applied[0]

        self.control_history.append([steer])
        logger.debug('steer: %.2f' % steer)
        t = time.time()
        logger.debug('dt=%.2f' % (t-self.last_cycle_time))
        self.last_cycle_time = t
        return applied


DEFAULT_CAR_THROTTLE = .36
DEFAULT_TRUCK_THROTTLE = 0.42
DEFAULT_BOAT_THROTTLE = 0.8
TOO_FEW_POINTS_MAXITER_DEFAULT = 1000
DEFAULT_MODE = 'carmission'
DEFAULT_DRY_RUN = False
DEFAULT_DISABLE_THROTTLE = False


def main():

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--throttle', type=float, default=None) # 0.01 for debug else None
    parser.add_argument('--minimum_throttle', type=float, default=None)
    parser.add_argument('--show_plot', action='store_true', default=False)
    parser.add_argument('--save_plot', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE)
    parser.add_argument('--error_kind', type=str, default=None)
   
    parser.add_argument('--steer_limit', type=float, default=None)
  
    parser.add_argument('--pid_tuning', type=str, default=None)
    parser.add_argument('--dont_randomize_dest_on_stopiteration', action='store_true', default=True)
    parser.add_argument('--throttle_disabled', action='store_true', default=DEFAULT_DISABLE_THROTTLE)
    parser.add_argument('--randomize_dest_every_n_seconds', type=float, default=None)
    args = parser.parse_args()
    logger.info('Got args: {}'.format(args))

    car = 'car' in args.mode.lower()
    truck = 'truck' in args.mode.lower()
    wheeled = car or truck
    boat = 'boat' in args.mode.lower()
    mission= 'mission' in args.mode.lower()
    chase = 'chase' in args.mode.lower()
    race = args.mode.lower() == 'race'
    boatrace = boat and race
    if args.throttle is None:
        args.throttle = (
            DEFAULT_TRUCK_THROTTLE if truck
            else DEFAULT_CAR_THROTTLE if (car or race) 
            else DEFAULT_BOAT_THROTTLE if boat 
            else min(DEFAULT_BOAT_THROTTLE, DEFAULT_CAR_THROTTLE, .4))
        logger.info('args.throttle: {}'.format(args.throttle))
    if args.error_kind is None: args.error_kind = 'cte' if (car or mission or chase) else 'heading'
    if args.steer_limit is None: args.steer_limit = .5 if (car or race or chase) else 1.0
    if args.pid_tuning is None:
        args.pid_tuning = 'heavy' if truck else 'chase' if chase else 'car' if car else 'boat'
    filters_present = ''
    if car:
        filters_present += 'car'
    if boat:
        filters_present += 'boat'
    if truck:
        filters_present += 'heavy'
    if mission:
        filters_present += 'mission'

    logger.info('pid_tuning: {}'.format(args.pid_tuning))

    logger.info('filters_present: {}'.format(filters_present))
    
    controller = IOController(
        throttle=args.throttle, 
        error_kind=args.error_kind, 
        pretuned=args.pid_tuning,
        filters_present=filters_present,
        do_perspective_transform=not boat,
        steer_limit=args.steer_limit,
        minimum_throttle=args.minimum_throttle,
        dry_run=DEFAULT_DRY_RUN,
        throttle_disabled=args.throttle_disabled,
    )

    last_gps_randomization_time = time.time()


    while True:
        try:
            do_randomize = False
            try:
                controller.step()

            except StopIteration as e:
                if args.dont_randomize_dest_on_stopiteration:
                    raise e
                else:
                    if controller.n_randomizations > controller._max_randomizations:
                        raise e
                    else:
                        do_randomize = True

            if controller.has_few_points:
                if controller.n_randomizations > controller._max_randomizations:
                    raise StopIteration('Too many GPS randomizations.')
                elif do_randomize:
                    # The idea is that, once we get to our destination, rather than just exiting,
                    # we shouldw select a new destination at random.
                    controller.randomize_gps()

            if args.randomize_dest_every_n_seconds is not None:
                assert args.randomize_dest_every_n_seconds > 0
                if time.time() - last_gps_randomization_time > args.randomize_dest_every_n_seconds:
                    last_gps_randomization_time = time.time()
                    controller.randomize_gps()
            
        except BaseException as e:

            if isinstance(e, KeyboardInterrupt):
                logger.info('User interrupted.')
                break

            elif isinstance(e, StopIteration):
                logger.info('StopIteration.')
                break

            else:
                from traceback import format_exc
                tb = format_exc()
                logger.warning('Stopping: %s' % tb)
                break

        if args.save_plot or args.show_plot:
            controller.stop()
            controller.plot(save=args.save_plot)
            if args.show_plot:
                plt.show()
   

if __name__ == '__main__':
    main()
