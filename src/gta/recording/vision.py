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
from PIL import Image

import wx

class Grab_wx:
    def __init__(self):
        self.app = wx.App()  # Need to create an App instance before doing anything

    def __call__(self):
        screen = wx.ScreenDC()
        size = screen.GetSize()
        bmp = wx.EmptyBitmap(size[0], size[1])
        mem = wx.MemoryDC(bmp)
        mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
        del mem  # Release bitmap
        return bmp

def win32_grab(hwnd, rect):
    import win32gui
    import win32ui 

    left, top, right, bot = rect
    w = right - left
    h = bot - top
    
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(save_bitmap)

    save_dc.BitBlt((0,0),(w, h), mfc_dc, (0,0), win32con.SRCCOPY)

    windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
    bmpinfo = save_bitmap.GetInfo()
    bmpstr = save_bitmap.GetBitmapBits(True)

    # This creates an Image object from Pillow
    bmp = Image.frombuffer('RGB',
                            (bmpinfo['bmWidth'],
                            bmpinfo['bmHeight']),
                            bmpstr, 'raw', 'BGRX', 0, 1)

    # save_bitmap.SaveBitmapFile(save_dc, bmpfilenamename)
    

    # Free Resources
    mfc_dc.DeleteDC()
    save_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    win32gui.DeleteObject(save_bitmap.GetHandle())

    return bmp

def win32_grab2(hwnd, rect):
    import win32gui
    import win32ui 
    
    left, top, right, bot = rect
    w = right - left
    h = bot - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(save_bitmap)

    windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
    bmpinfo = save_bitmap.GetInfo()
    bmpstr = save_bitmap.GetBitmapBits(True)

    # This creates an Image object from Pillow
    bmp = Image.frombuffer('RGB',
                            (bmpinfo['bmWidth'],
                            bmpinfo['bmHeight']),
                            bmpstr, 'raw', 'BGRX', 0, 1)
    # bmp.save("asdf.png")

    # Free Resources
    mfc_dc.DeleteDC()
    save_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    win32gui.DeleteObject(save_bitmap.GetHandle())

    return bmp

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
    print(bbox, [width, height])
    return grab_pyautogui(region=(from_left, from_top, width, height))
    im = grab_pyautogui()
    return np.array(im)[from_top:from_bottom, from_left:from_right]


class Window(object):
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
        
    def grab(self, bbox=None, relativeBbox=None):
        # start = time.time()
        if bbox is None: bbox = self.getBbox()
        if relativeBbox is not None:
            bbox = (np.array(bbox) + relativeBbox).tolist()
        if False:
            out =  grab_pyautogui_bbox(bbox)
        else:
            from PIL import ImageGrab, Image
            out = ImageGrab.grab(bbox, all_screens=True)
        # print('Grabbed in', time.time() - start, 'sec.')
        return out

    # Not sure what the rest of this is for.

    def _window_enum_callback(self, hwnd, wildcard):
        '''Pass to win32gui.EnumWindows() to check all the opened windows'''
        if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) is not None:
            self._hwnd = hwnd

    def find_window_wildcard(self, wildcard):
        self._hwnd = None
        win32gui.EnumWindows(self._window_enum_callback, wildcard)

    def kill_task_manager(self):
        wildcard = 'Gestionnaire des t.+ches de Windows'
        self.find_window_wildcard(wildcard)
        if self._hwnd:
            win32gui.PostMessage(self._hwnd, win32con.WM_CLOSE, 0, 0)
            time.sleep(0.5)


class GtaWindow(Window):

    def __init__(self, fpath=None):
        wids = []
        def saveWid(wid, *args):
            wids.append(wid)
        win32gui.EnumWindows(saveWid, None)

        widsByTitle = {win32gui.GetWindowText(wid): wid for wid in wids}
        gtaWid = widsByTitle['Grand Theft Auto V']
        
        Window.__init__(self, gtaWid)

        self.window_size = self.grab().size[:2]
        print('Window size:', self.window_size)

        # Make second number bigger if we tend to go into the right shoulder.
        self.car_origin = self.wscale(45, 48)
        self.car_origin_minimap = self.wscale(100, 110)
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
    def minimap_perspective_transform(img):
        from_points = np.array([
            (85, 70),
            (140, 70),
            (150, 108),
            (75, 108),
        ]) # x, y coordinates

        lined = cv2.polylines(np.array(img), [np.array(from_points)], True, (255, 255, 0), 1)
        cv2.imshow('Before Perspective Transform', lined)

        # TODO: The folowing points are backwards (y,x) and need to be tuned.
        to_points = [
            (55, 115),
            (85, 115),
            (85, 200),
            (55, 200),
        ]

        M = cv2.getPerspectiveTransform(from_points, to_points)
        warped = cv2.warpPerspective(img, M)#, (maxWidth, maxHeight))
        cv2.imshow('Perspective Warped', warped, img.shape[:2])
        cv2.waitKey(0)

    @property
    def minimap(self):
        return self.grab(relativeBbox=self.wscale(35, 588, -758, -38))

    @property
    def micromap(self):
        return self.grab(relativeBbox=self.wscale(100, 640, -820, -52))

    @property
    def track_mask(self):
        return self.get_track_mask()

    def get_track_mask(self, basemap_kind='micromap', do_erode=True):
        if basemap_kind == 'micromap':
            basemap = np.array(self.micromap)
        else:
            assert basemap_kind == 'minimap'
            basemap = np.array(self.minimap)
        # hsv = cv2.cvtColor(basemap, cv2.COLOR_RGB2HSV)
        # r = basemap[..., 0]
        # g = basemap[..., 1]
        # b = basemap[..., 2]

        # s = hsv[..., 1]

        from functools import reduce
        AND = np.logical_and
        OR = lambda *args: reduce(np.logical_or, args)

        # cv2.imshow('tm', basemap)
        # cv2.waitKey(0)

        def RGB(r, g, b, radius=5):
            lower = np.array([r-5, g-5, b-5])
            upper = np.array([r+5, g+5, b+5])
            return cv2.inRange(basemap, lower, upper)

        out = OR(
            # RGB(162, 29, 89, radius=10), # purple waypoint tick
            RGB(79,5,154),
            RGB(161,73,239),
            RGB(168, 84, 243), # magenta line
            # RGB(240, 200, 90), # yellow line
            RGB(240,200,80), # Race dots
        ).astype('uint8')

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
        print('Got image of shape', img.shape, '.')
        return img


class VisionRecorder(BaseRecorder):

    def __init__(self, period=.01, **taskConstructorKwargs):
        self.gtaWindow = GtaWindow()
        super(self.__class__, self).__init__(period, Task=VisionTask, **taskConstructorKwargs)


if __name__ == '__main__':
    window = GtaWindow()
    window.minimap_perspective_transform(window.minimap)
