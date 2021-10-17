import logging

logformat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
datefmt = '%H:%M:%S'
logging.basicConfig(format=logformat, datefmt=datefmt, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time

# Make program aware of DPI scaling
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

from os.path import dirname, join
import sys
import os
here = dirname(__file__)
if os.name == 'nt':
    # VSCode does some shenanigans https://github.com/microsoft/vscode-python/issues/13811
    here = here.replace('/', '\\')
sys.path.append(join(here, '..', '..'))

#From this point on calls like GetWindowRect() should return the proper values.

import win32gui
import win32com
import win32com.client
import win32con
import numpy as np

from gta.recording import BaseRecorder, BaseTask
import cv2
from PIL import ImageGrab, Image

import pydirectinput



class WXGrabber:
    def __init__(self):
        import wx
        self.app = wx.App()  # Need to create an App instance before doing anything

    def __call__(self, bbox=None):
        import wx
        screen = wx.ScreenDC()

        if bbox is None:
            size = screen.GetSize()
            bbox = 0, 0, size[0], size[1]

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        bmp = wx.EmptyBitmap(width, height)
        mem = wx.MemoryDC(bmp)
        #https://wxpython.org/Phoenix/docs/html/wx.DC.html#wx.DC.Blit
        mem.Blit(0, 0, width, height, screen, x1, y1)
        del mem  # Release bitmap

        img = bmp.ConvertToImage()
        buf = img.GetDataBuffer() # use img.GetAlphaBuffer() for alpha data
        arr = np.frombuffer(buf, dtype='uint8')
        arr = arr.reshape([height, width, 3])
        return Image.fromarray(arr)



def grab_pyautogui(region=None):
    import pyautogui
    if region is not None:
        return pyautogui.screenshot(region='region')
    else:
        return pyautogui.screenshot()


def grab_pyautogui_bbox(bbox):
    from_left, from_top, from_right, from_bottom = bbox
    width = from_right - from_left
    height = from_bottom - from_top
    return grab_pyautogui(region=(from_left, from_top, width, height))


class D3DGrabber:

    def __init__(self):
        import d3dshot
        self.d = d3dshot.create(capture_output="numpy")
        self.d.capture()
        time.sleep(4)

    def __call__(self, bbox):
        left, top, right, bot = bbox
        frame = self.d.get_latest_frame()
        out = frame[top:bot, left:right, :]
        return out

    def __del__(self):
        self.d.stop()

    
# Done by Frannecklp

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

# class Win32Grabber:

#     def __init__(self):
        
def grab_screen_win32(region=None):

    hwin = win32gui.GetDesktopWindow()

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()

    if region:
        left,top,x2,y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            

class Window:
    def __init__(self, wid):
        self._hwnd = wid
        self.shell = win32com.client.Dispatch("WScript.Shell")

    def BringToTop(self):
        win32gui.BringWindowToTop(self._hwnd)

    def SetAsForegroundWindow(self):
        self.shell.SendKeys('%')
        win32gui.SetForegroundWindow(self._hwnd)

    def Maximize(self):
        win32gui.ShowWindow(self._hwnd, win32con.SW_MAXIMIZE)

    def setActWin(self):
        win32gui.SetActiveWindow(self._hwnd)
        
    def getBbox(self):
        a, b, c, d = win32gui.GetWindowRect(self._hwnd)
        return a+3, b+32, c-3, d-3
        # Experimentally found for windowed game at game size 1024x768, without hidpi scaling trick
        # return a+35, b+80, c+235, d + 203
        
    def grab(self, bbox=None, relativeBbox=None, kind='ndarray'):
        if not hasattr(self, '_last_fullscreen_grab'):
            self.reset_fullscreen_grab()
        
        out = self._last_fullscreen_grab

        full_height, full_width = out.shape[:2]

        assert bbox is None
        bbox = np.array([0, 0, full_width, full_height])
        if relativeBbox is None:
            relativeBbox = np.array([0, 0, 0, 0])

        left, top, right, bot = bbox = bbox + relativeBbox
        assert left < right
        assert top < bot  # origin is top left

        out = out[top:bot, left:right, ...]

        if kind == 'PIL' and not isinstance(out, Image.Image):
            # TODO: According to the README, d3dshot is even faster if you use numpy output (which we do, but then we convert to PIL image here). Try using an all-ndarray pipeline if possible.
            if isinstance(out, np.ndarray):
                print('WARNING: CONVERTING IS SLOWER')
                out = Image.fromarray(out)
        elif kind == 'ndarray' and not isinstance(out, np.ndarray):
            if not isinstance(out, np.ndarray):
                print('WARNING: CONVERTING IS SLOWER')
                out = np.array(out)

        return out

    def reset_fullscreen_grab(self):
        self._last_fullscreen_grab = self._grab()
        
    def _grab(self, bbox=None, relativeBbox=None):

        if bbox is None: bbox = self.getBbox()
        
        if relativeBbox is not None:
            bbox = (np.array(bbox) + relativeBbox).tolist()

        if False:
            out =  grab_pyautogui_bbox(bbox)
        elif False:
            if not hasattr(self, 'grabber'): self.grabber = D3DGrabber()
            out = self.grabber(bbox)
        elif True:
            out = grab_screen_win32(bbox)
        else:
            out = ImageGrab.grab(bbox, all_screens=True)
            
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        return out

    def sendKeys(self, keys, delay=0.1):
        """Send specified keys to the saved hwnd."""
        for k in keys:
            pydirectinput.keyDown(k)
            time.sleep(delay)
            pydirectinput.keyUp(k)
        

class GtaWindow(Window):

    def __init__(self):
        wids = []
        def saveWid(wid, *unused_args):
            wids.append(wid)
        win32gui.EnumWindows(saveWid, None)

        gta_wids = [
            wid for wid in wids
            if 'Grand Theft Auto V' in win32gui.GetWindowText(wid)
        ]
        gta_wid = gta_wids[-1]
        if len(gta_wids) > 1:
            msg = []
            msg.append('Multiple GTA windows found:')
            for gta_wid_ in gta_wids:
                msg.append('  {}'.format(win32gui.GetWindowText(gta_wid_)))
            msg.append('Using WID: {}'.format(gta_wid))
            msg.append('If this leads to errors, close or rename the other windows and try again.')
            logger.warning('\n'.join(msg))
        
        Window.__init__(self, gta_wid)

        self.window_size = self.grab(kind='numpy').shape[:2][::-1]
        logger.info('Window size: {}'.format(self.window_size))
        umap = self.micromap
        logger.info('Micro map size: {}'.format(umap.shape))

        # Make second number bigger if we tend to go into the right shoulder.
        self.car_origin = self.wscale(45, 47)
        self.car_origin_micromap_perspectiveTransformed = self.wscale(25, 17.5)
        self.car_origin_minimap = self.wscale(100, 110)
        self.car_origin_minimap_perspectivetransformed = self.wscale(100, 60)  # TODO: Retune these values.
        self.last_cycle_time = time.time()

    def wscale(self, a, b, c=None, d=None):
        w1 = float(self.window_size[1]) / 768.
        w2 = float(self.window_size[0]) / 1024.
        out = [
            type(a)(w1 * a), type(b)(w2 * b)
        ]

        if c is not None and d is not None:
            out.extend([
                type(c)(w1 * c), type(d)(w2 * d)
            ])

        return tuple(out)

    @property
    def img(self):
        return self.grab()

    @staticmethod
    def minimap_perspective_transform(img, visualize=False):
        from_points = np.array([
            (93, 52),
            (166, 52),
            (166, 80),
            (75, 80),
        ], dtype='float32') # x, y coordinates (or column, row)

        if visualize:
            line_to_draw = np.array(from_points.reshape((-1, 1, 2)), dtype='int32')
            lined = cv2.polylines(np.array(img), line_to_draw, True, (255, 255, 0), 8)
            cv2.imshow('Before Perspective Transform', lined)

        d = 45
        x0 = 145
        y0 = 196
        to_points = np.array([
            (x0,   y0),
            (x0+d, y0),
            (x0+d, y0+d),
            (x0,   y0+d),
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(from_points, to_points)
        h, w = img.shape[:2]
        logger.debug('Old size: {}x{}'.format(w, h))
        w = int(w * 1.15)
        h = int(h * 1.5)
        logger.debug('New size: {}x{}'.format(w, h))
        warped = cv2.warpPerspective(img, M, (w, h))

        if visualize:
            lined2 = cv2.polylines(np.array(warped), np.array(to_points.reshape((-1, 1, 2)), dtype='int32'), True, (255, 0, 255), 8)
            cv2.imshow('Perspective Warped', lined2)
            cv2.waitKey(1)

        return warped

    @property
    def minimap(self):
        return self.grab(relativeBbox=self.wscale(35, 588, -758, -38))

    @property
    def micromap(self):
        return self.grab(relativeBbox=self.wscale(100, 640, -820, -52))

    @property
    def track_mask(self):
        return self.get_track_mask()

    @staticmethod
    def perspective_minimap_to_micromap(minimap):
        DY, DX = minimap.shape[:2]
        dx1 = int(0.4 * DX)
        dx2 = int(0.45 * DX)
        dy1 = int(0.8 * DY)
        dy2 = int(0.05 * DY)
        assert dx1 + dx2 < DX
        assert dy1 + dy2 < DY
        return minimap[dy1:-dy2, dx1:-dx2]

    def get_track_mask(self, basemap_kind='micromap', do_erode=True, filters_present=['all'], do_perspective_transform=True):
        
        t1 = time.time()
        if basemap_kind == 'minimap' or do_perspective_transform:
            basemap = np.array(self.minimap)
        else:
            basemap = np.array(self.micromap)
            assert basemap_kind == 'micromap'
        t2 = time.time()

        if do_perspective_transform:
            basemap = self.minimap_perspective_transform(basemap)

            if basemap_kind == 'micromap':
                basemap = self.perspective_minimap_to_micromap(basemap)

        # hsv = cv2.cvtColor(basemap, cv2.COLOR_RGB2HSV)
        # r = basemap[..., 0]
        # g = basemap[..., 1]
        # b = basemap[..., 2]

        # s = hsv[..., 1]

        from functools import reduce
        AND = np.logical_and
        OR = lambda *args: reduce(np.logical_or, args)


        def RGB(r, g, b, radius=5):
            lower = np.array([r-5, g-5, b-5])
            upper = np.array([r+5, g+5, b+5])
            return cv2.inRange(basemap, lower, upper)

        filters = dict(
            magenta_line=(168, 84, 243),
            yellow_line=(241, 203, 88),
            green_line=(121, 206, 121),
            sky_line=(101, 185, 230),
            race_dots=(240, 200, 80),
            purple_cross=(161, 73, 239),
            bluish=(79, 5, 154),
        )

        if 'all' in filters_present:
            filters_present = filters.keys()

        out = []
        for filter in filters_present:
            out.append(RGB(*filters[filter]))

        out = OR(*out).astype('uint8')
        #     # RGB(162, 29, 89, radius=10), # purple waypoint tick
        #     RGB(79,5,154),
        #     RGB(161,73,239),
        #     RGB(168, 84, 243), # magenta line
        #     # RGB(240, 200, 90), # yellow line
        #     RGB(240,200,80), # Race dots
        # ).astype('uint8')

        if do_erode:
            out = cv2.erode(out, np.ones((3, 3)))



        return out.astype('bool'), basemap


class VisionTask(BaseTask):

    def __init__(self, minimap=False):
        self.minimap = minimap
        self.window = GtaWindow()

    def __call__(self):
        if self.minimap:
            img = self.window.minimap
        else:
            img = self.window.img
        img = np.array(img)
        logger.debug('Got image of shape {}.'.format(img.shape))
        return img


class VisionRecorder(BaseRecorder):

    def __init__(self, period=.01, **taskConstructorKwargs):
        self.gtaWindow = GtaWindow()
        super(self.__class__, self).__init__(period, Task=VisionTask, **taskConstructorKwargs)


if __name__ == '__main__':
    window = GtaWindow()
    window.minimap_perspective_transform(window.minimap, visualize=True)
