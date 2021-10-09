from os.path import join, dirname
HERE = dirname(__file__)
from sys import path
path.append(join(HERE, '..'))

import gta.read_entitystate_data as red
from glob import glob
import gc

from tqdm.auto import tqdm

if __name__ == '__main__':
    binfiles = glob(join(red.DATA_DIR, '*.bin'))

    for binfile in tqdm(binfiles, unit='binfile'):
        print('Loading {}'.format(binfile))
        # tracks = red.read_data(binfile)
        # tm = red.TrackManager(tracks)
        tm = red.TrackManager(binfile)
        print('Loaded; unloading.')

        # del tracks
        del tm
        gc.collect()
