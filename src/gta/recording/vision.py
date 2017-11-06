import win32gui, win32com, win32com.client
import numpy as np

from gta.recording import BaseRecorder, BaseTask


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
        return a + 3, b + 26, c - 3, d - 3 # Remove Windows chrome.
    
    def grab(self, bbox=None, relativeBbox=None):
        from PIL import ImageGrab, Image
        if bbox is None: bbox = self.getBbox()
        if relativeBbox is not None:
            bbox = (np.array(bbox) + relativeBbox).tolist()
        return ImageGrab.grab(bbox)

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

    @property
    def img(self):
        return self.grab()

    @property
    def minimap(self):
        return self.grab(relativeBbox=[57, 589, -1092, -33])


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

