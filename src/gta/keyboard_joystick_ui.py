#!/usr/bin/env python

from sys import stdout
import time
import gta.gameInputs.gamepad
import msvcrt


if __name__ == '__main__':
    gpad = gta.gameInputs.gamepad.Gamepad()

    walk_time = 1.0
    drive_time = 1.2
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

    def accel():
        gpad(accel=1)
        time.sleep(drive_time)
        gpad(accel=0)

    def decel():
        gpad(decel=1)
        time.sleep(drive_time)
        gpad(decel=0)


    print('''
    W: forward
    S: backward
    A: left strafe
    D: right strafe
    Q: left turn
    E: right turn
    F: accel
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
