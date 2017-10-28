import vjoy

class Axis(object):
    
    def __init__(self, vjoyName, inRange=(-.5, .5), outRange=(0, 32000), neutralInValue=0):
        self.vjoyName = vjoyName
        self.inRange = inRange
        self.outRange = outRange
        self.neutralInValue = neutralInValue
        self.outValue = self(neutralInValue)
        
    def __call__(self, x):
        x = min(max(x, self.inRange[0]), self.inRange[1])
        #assert self.inRange[0] <= x <= self.inRange[1], x
        din = self.inRange[1] - self.inRange[0]
        dout = self.outRange[1] - self.outRange[0]
        outFloat = (x - self.inRange[0]) * dout / din + self.outRange[0]
        return type(self.outRange[0])(outFloat)
    
    def setValue(self, x):
        outValue = self(x)
        self.outValue = outValue


# class Button(object):

#     def __init__(self, vjoyName, neutralInValue=False):
#         self.vjoyName = vjoyName
#         self.outValue = neutralInValue

#     def setValue(self, x):
#         self.outValue = x

   
class JoystickEmulator(object):
    
    def __init__(self):
        self.vj = vjoy.vj
        self.axes = dict(
            lx=Axis('wAxisX'),
            ly=Axis('wAxisY'),
            rx=Axis('wAxisZ'),
            ry=Axis('wAxisXRot'),
            lt=Axis('wSlider', (0, 1.)),
            rt=Axis('wDial', (0, 1.)),
        )
        def do(action):
            def f(*args, **kwargs):
                out = action(*args, **kwargs)
                self.update()
                return out
            return f
        self.accel = do(self.axes['rt'].setValue)
        self.decel = do(self.axes['lt'].setValue)
        self.yaw = do(self.axes['lx'].setValue)
        self.pitch = do(self.axes['ly'].setValue)
        
    def update(self):
        position = self.vj.generateJoystickPosition(**{
            axis.vjoyName: axis.outValue
            for axis in self.axes.values()
        })
        self.vj.update(position)
        
    def neutral(self):
        for axis in self.axes.values():
            axis.setValue(axis.neutralInValue)
        self.update()