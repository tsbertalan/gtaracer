#!/usr/bin/env python

import time
import gta.gameInputs.gamepad
import msvcrt

if __name__ == '__main__':
    gpad = gta.gameInputs.gamepad.Gamepad()

    walk_time = 1.0
    turn_time = 0.1

    def forward():
        gpad(walk=1)
        time.sleep(walk_time)
        gpad(walk=0)

    def backward():
        gpad(walk=-1)
        time.sleep(walk_time)
        gpad(walk=0)

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

    print('''
    W: forward
    S: backward
    A: left strafe
    D: right strafe
    Q: left turn
    E: right turn
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
        else:
            print('unknown key:', key, '; quit.')
            break
