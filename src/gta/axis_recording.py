from time import sleep
from sys import stdout

sleep(10)

print('Initialize vjoy')
from gta.gameInputs.gamepad import *

vj = vjoy.vJoy()
print('perturbing')


# axis defauts:
'''
        # left thb x        left thb y     left trigger
        wAxisX = 16393,   wAxisY = 16393,   wAxisZ = 0,

        # right thb x       right thb y        right trigger
        wAxisXRot = 16393, wAxisYRot = 16393, wAxisZRot = 0,

        # ???         ???        ???
        wSlider = 0, wDial = 0, wWheel = 0,
        # ???         ???        ???
        wAxisVX = 0, wAxisVY = 0, wAxisVZ = 0,
        # ???         ???                ???                   
        wAxisVBRX = 0, wAxisVBRY = 0, wAxisVBRZ = 0,
'''

# for axname in 'wAxisX', 'wAxisY', 'wAxisZ', 'wAxisXRot', 'wAxisYRot', 'wAxisZRot', 'wSlider', 'wDial', 'wWheel', 'wAxisVX', 'wAxisVY', 'wAxisVZ', 'wAxisVBRX', 'wAxisVBRY', 'wAxisVBRZ':
for axname in 'wSlider',:

    print('axis=%s:' % axname, end=' ')

    ax = Axis(axname, neutral=None, nominal_range=(-1, 1), input_clamp=[-1, 1])

    def go_to_one():
        positions = {}
        positions.update(ax(1))
        joypos = vj.generateJoystickPosition(**positions)
        vj.update(joypos)

    def go_to_zero():
        print(0, end='')
        stdout.flush()
        positions = {}
        positions.update(ax(0))
        joypos = vj.generateJoystickPosition(**positions)
        vj.update(joypos)

    for rep in range(4):
        go_to_one()
        sleep(.5)
        go_to_zero()
        sleep(.5)
    print()

    sleep(1)




print('exit')