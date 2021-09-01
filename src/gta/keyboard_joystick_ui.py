#!/usr/bin/env python

from sys import stdout
import time
from sys import path
from os.path import join, expanduser
path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))
import gta.gameInputs.gamepad
import msvcrt


if __name__ == '__main__':
    gpad = gta.gameInputs.gamepad.Gamepad()

    walk_time = 1.0
    drive_time = 1.2
    turn_time = 0.1


    def forward():
        if not forward.current_walk_speed == 1:
            gpad(walk=1)
            forward.current_walk_speed = 1
        else:
            gpad(walk=0)
            forward.current_walk_speed = 0
    forward.current_walk_speed = 0

    def backward():
        if not forward.current_walk_speed == -1:
            gpad(walk=-1)
            forward.current_walk_speed = -1
        else:
            gpad(walk=0)
            forward.current_walk_speed = 0
            
    def leftStrafe():
        gpad(steer=-1)
        time.sleep(walk_time)
        gpad(steer=0)

    def rightStrafe():
        gpad(steer=1)
        time.sleep(walk_time)
        gpad(steer=0)

    def leftTurn():
        gpad(walkturn=-1)   
        time.sleep(turn_time)
        gpad(walkturn=0)

    def rightTurn():
        gpad(walkturn=1)
        time.sleep(turn_time)
        gpad(walkturn=0)

    max_drive_speed = .5

    def accel():
        if not accel.current_drive_speed == max_drive_speed:
            gpad(accel=max_drive_speed, decel=0)
            accel.current_drive_speed = max_drive_speed
        else:
            gpad(accel=0, decel=0)
            accel.current_drive_speed = 0
    accel.current_drive_speed = 0

    def decel():
        if not accel.current_drive_speed == -max_drive_speed:
            gpad(accel=0, decel=max_drive_speed)
            accel.current_drive_speed = -max_drive_speed
        else:
            gpad(accel=0, decel=0)
            accel.current_drive_speed = 0


    print('''
    W: forward (walk)
    S: backward (walk)
    A: left strafe (walk) or turn (drive)
    D: right strafe (walk) or turn (drive)
    Q: left turn (walk)
    E: right turn (walk)
    F: accel (drive) or shoot (walk)
    R: decel
    Other keys: quit
    ''')

    # Read WASD keys from terminal in a loop.
    while True:
        key = msvcrt.getch()
        # Convert bytestring to string.
        key = key.decode("utf-8")

        if key == 'w':
            forward()
        elif key == 'a':
            leftStrafe()
        elif key == 's':
            backward()
        elif key == 'd':
            rightStrafe()
        elif key == 'q':
            leftTurn()
        elif key == 'e':
            rightTurn()
        elif key == 'f':
            accel()
        elif key == 'r':
            decel()
        else:
            print('quit')
            break
        print('.', end='')
        stdout.flush()
