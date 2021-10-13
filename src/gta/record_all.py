from sys import path
import time
from tqdm.auto import tqdm
from os.path import expanduser, join


path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))

import time
from argparse import ArgumentParser
from gta.recording.unified import UnifiedRecorder
import gta.default_configs


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--record_path', type=str, default=None)
    parser.add_argument('--record_time', type=int, default=50)
    args = parser.parse_args()
    if args.record_path is None:
        datename = '%s-gtav_recording.npz' % time.strftime('%Y-%m-%d-%H-%M-%S')
        args.record_path = join(gta.default_configs.SCREEN_CONTROLLER_REC_DIR, datename)

    # Record everything with the unified recorder.
    print('Recording to %s' % args.record_path)
    recorder = UnifiedRecorder(gamepadPeriod=0.01)
    recorder.create_subprocesses()
    recorder.start()

    for sec in tqdm(range(args.record_time), desc='Recording', unit='second'):
        time.sleep(1)

    print('Saving data to %s' % args.record_path)
    recorder.stop()
    recorder.save(args.record_path)
