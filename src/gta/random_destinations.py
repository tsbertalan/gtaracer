"""Send keyboard inputs to select a random destination from the interaction menu.

Only works when the game is focused. :("""

import time
from random import uniform

import pydirectinput

def change_gps(gp):

    def back():
        button = gp.named_buttons['back']
        button.tap(duration=.5)

    def press(key, duration=.05):
        
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
    
    # Open the interaction menu.
    back()

    # Go down one (assuming we're a CEO or MC present).
    press('down', duration=.1)

    # Holding left for a while.
    left_time = uniform(.05, 1.0)
    press('left', duration=left_time)

    # Select the option.
    press('enter')


if __name__ == '__main__':
    from tqdm.auto import tqdm
    from sys import path
    from os.path import join, expanduser
    path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))
    import gta.gameInputs.gamepad
    gpad = gta.gameInputs.gamepad.Gamepad()
    print('Focus the GTA window now.')
    wait = 10
    for sec in tqdm(range(wait), desc='Waiting %s seconds' % wait):
        time.sleep(1)
    change_gps(gpad)
