from sys import path
from os.path import expanduser, join


path.append(join(expanduser('~'), 'Dropbox', 'Projects', 'GTARacer', 'src'))

import time
from argparse import ArgumentParser
from gta.recording.unified import UnifiedRecorder


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--record_path', type=str, default='recordings/test.npz')
    args = parser.parse_args()

    # Record everything with the unified recorder.
    print('Recording to %s' % args.record_path)
    recorder = UnifiedRecorder()
    recorder.create_subprocesses()
    recorder.start()

    time.sleep(100)

    # try:
    #     while True:
    #         time.sleep(1)
        
    # except KeyboardInterrupt:
    #     print('Stopping recording...')

    print('Saving data to %s' % args.record_path)
    recorder.stop()
    recorder.save(args.record_path)
