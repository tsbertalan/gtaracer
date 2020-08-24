import time

# Make program aware of DPI scaling
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()
#From this point on calls like GetWindowRect() should return the proper values.

import win32gui
import win32com
import win32com.client
import win32con
import numpy as np

from gta.recording import BaseRecorder, BaseTask
import cv2
from PIL import Image

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
        start = time.time()
        from PIL import ImageGrab, Image
        if bbox is None: bbox = self.getBbox()
        if relativeBbox is not None:
            bbox = (np.array(bbox) + relativeBbox).tolist()
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
        
        #self.widsByTitle = widsByTitle

        Window.__init__(self, gtaWid)

        self.car_origin = 49, 46 # Make second number bigger if we tend to go into the right shoulder.
        self.last_cycle_time = time.time()

    @property
    def img(self):
        return self.grab()

    @property
    def minimap(self):
        return self.grab(relativeBbox=[35, 588, -758, -38])

    @property
    def micromap(self):
        return self.grab(relativeBbox=[100, 640, -820, -52])

    @property
    def track_mask(self):
        micromap = np.array(self.micromap)
        # hsv = cv2.cvtColor(micromap, cv2.COLOR_RGB2HSV)
        # r = micromap[..., 0]
        # g = micromap[..., 1]
        # b = micromap[..., 2]

        # s = hsv[..., 1]

        AND = np.logical_and
        OR = np.logical_or

        lower_magenta = np.array([163, 79, 238])
        upper_magenta = np.array([173, 89, 248])

        lower_yellow = np.array([230, 190, 70])
        upper_yellow = np.array([250, 210, 90])

        return (
            cv2.erode(
                # AND(s > 80, AND(r > 60, b > 50)).astype('uint8') * 255,
                OR(
                    cv2.inRange(micromap, lower_magenta, upper_magenta),
                    cv2.inRange(micromap, lower_yellow, upper_yellow),
                ).astype('uint8')
                ,
                np.ones((3, 3))
            )
        ).astype('bool')


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

