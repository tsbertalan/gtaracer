from os.path import expanduser, join, basename, dirname, exists
import json
HERE = dirname(__file__)
HOME = expanduser('~')
from glob import glob
from datetime import datetime
import numpy as np
import cv2, PIL
import matplotlib.pyplot as plt
try:
    from . import read_entitystate_data
except ImportError:
    import read_entitystate_data
from tqdm.auto import tqdm

from typing import Iterator, Optional
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation


try:
    from .train_velocity_predictor import VELOCITY_DATA_DIR
except ImportError:
    from train_velocity_predictor import VELOCITY_DATA_DIR


class ImageRecording:

    def __init__(self, fname):
        self.fname = fname
        
    @property
    def fname_date(self):
        b = basename(self.fname)
        date_part = '-'.join(b.split('-')[:6])
        return datetime.strptime(date_part, '%Y-%m-%d-%H-%M-%S')

    def __str__(self) -> str:
        return '%s("%s")' % (
            type(self).__name__,
            self.fname
        )

    def unload_data(self):
        if hasattr(self, '_data'):
            print('Unloading %s' % self)
            if hasattr(self, '_gamepad_events'):
                del self._gamepad_events
            if hasattr(self, '_keyboard_events'):
                del self._keyboard_events
            del self._times
            del self._images
            del self._data

            # Trigger garbage collection.
            import gc
            gc.collect()

    def load_data_idempotent(self):
        if not hasattr(self, '_data') or not hasattr(self, '_times'):
            print('Loading %s' % self)
            self._data = np.load(self.fname, allow_pickle=True)
            if len(self._data['Y']) == 2:
                self._gamepad_events, self._keyboard_events = self._data['Y']
            else:
                assert len(self._data['Y']) == 1
                self._keyboard_events = self._data['Y'][0]
                self._gamepad_events = None
            self._times = []
            self._images = []
            for (t, img) in self._data['X']:
                self._times.append(t)
                self._images.append(img)
            
            self._times = np.array(self._times)

            # Make sure times are sorted.
            np.testing.assert_array_equal(np.sort(self._times), self._times)

        return True

    @property
    def times(self):
        self.load_data_idempotent()
        return self._times

    @property
    def images(self):
        self.load_data_idempotent()
        return self._images

    @property
    def gamepad_events(self):
        self.load_data_idempotent()
        return self._gamepad_events

    @property
    def keyboard_events(self):
        self.load_data_idempotent()
        return self._keyboard_events

    @property
    def tmin(self):
        return self.times[0]

    @property
    def tmax(self):
        return self.times[-1]


class TelemetryRecording:

    def __init__(self, fname):
        self.fname = fname

    def __str__(self) -> str:
        return '%s("%s")' % (
            type(self).__name__,
            self.fname
        )

    def load_data_idempotent(self):
        if not hasattr(self, '_track_manager'):
            #print('Loading %s' % self)
            self._track_manager = read_entitystate_data.TrackManager(self.fname)

        return True
        
    @property
    def track_manager(self):
        self.load_data_idempotent()
        return self._track_manager


class PairedRecording:

    def __init__(self, image_recording, telemetry_recordings):
        self.image_recording = image_recording
        self.telemetry_recordings = telemetry_recordings

    def plot_detections(self, image_index):
        image = self.image_recording.images[image_index]
        t = self.image_recording.times[image_index]
        track_managers = [trec.track_manager for trec in self.telemetry_recordings]
        return show_detections_on_image(
            image, t, track_managers,
        )

    def unload_image_data(self):
        self.image_recording.unload_data()

    @property
    def save_name_base(self):
        return self.image_recording.fname

    def write_video(self, **kwargs):
        fname = self.save_name_base + '.gif'
        print('Writing video to', fname)

        def get_images():
            tracks_seen = set()
            for i in tqdm(range(len(self.image_recording.images)), unit='images', desc='Creating video'):
                img, active = self.plot_detections(i)
                tracks_seen.update([
                    track 
                    for track in active
                    if
                    np.any([
                        sx != -1 for sx in 
                        track._get_data('screenx')
                    ])
                ])
                yield img
            print('A total of {} tracks were seen in recording {}.'.format(
                len(tracks_seen),
                basename(self.image_recording.fname),
            ))

        write_animation(
            [im for im in get_images()],
            fname, **kwargs
        )


def get_known_pairings(paired):
    return {
        basename(p.image_recording.fname): [basename(t.fname) for t in p.telemetry_recordings]
        for p in paired
    }


KNOWN_BASENAME = 'known.json'
def save_known_pairings(data, dir_path=None):
    known = get_known_pairings(data['paired'])
    if dir_path is None:
        dir_path = dirname(list(known.keys())[0])
    known_path = join(dir_path, KNOWN_BASENAME)
    with open(known_path, 'w') as fp:
        fp.write(json.dumps(known))


def find_filenames(base_dir=join(VELOCITY_DATA_DIR, 'Protocol V1'), check_for_existing=True, save_known=True, pair_with_truncated_recs=False) -> dict:

    known = {}
    if check_for_existing:
        known_path = join(base_dir, KNOWN_BASENAME)
        if exists(known_path):
            with open(known_path) as fp:
                known = json.load(fp)

    all_npz = glob(join(base_dir, '*.npz'), recursive=True)
    img_recs = [
        ImageRecording(fname) 
        for fname in all_npz 
        if not fname.endswith('reduced.npz') and not '_paired' in fname
    ]
    print('Found {} image recording(s) in {}.'.format(len(img_recs), base_dir))

    all_bin = glob(join(base_dir, '*.bin'), recursive=True)
    tel_recs = [TelemetryRecording(fname) for fname in all_bin]
    print('Found {} telemetry recording(s) in {}.'.format(len(tel_recs), base_dir))

    trunc_tel_recs = [rec for rec in tel_recs if 'trunc' in rec.fname]
    nontrunc_tel_recs = [rec for rec in tel_recs if rec not in trunc_tel_recs]

    paired_recordings = []
    LIMIT = 9999999999999
    telemetries = (trunc_tel_recs[:LIMIT] if pair_with_truncated_recs else nontrunc_tel_recs[:LIMIT])
    imagess = img_recs[:LIMIT]
    loaded_tel_recs = {basename(t.fname): t for t in tel_recs}
    loaded_telemetries_maybe_truncated = {basename(t.fname): t for t in telemetries}
    
    for images in tqdm(imagess, unit='npz file', desc='Pairing data files'):
        images_basename = basename(images.fname)
        if images_basename in known:
            print('Image archive %s is in %s.' % (images_basename, KNOWN_BASENAME))
            telemetries_for_images = []
            for fname in known[basename(images.fname)]:
                tel_basename = basename(fname)
                if tel_basename not in loaded_tel_recs:
                    loaded_tel_recs[tel_basename] = TelemetryRecording(fname)
                    if ('trunc' in fname and pair_with_truncated_recs) or ('trunc' not in fname and not pair_with_truncated_recs):
                        loaded_telemetries_maybe_truncated[tel_basename] = loaded_tel_recs[fname]
                telemetries_for_images.append(loaded_tel_recs[tel_basename])
        else:
            print('Image archive %s is not in %s.' % (images_basename, KNOWN_BASENAME))
            telemetries_for_images = []
            for telemetry in loaded_telemetries_maybe_truncated.values():
                try:
                    if images.tmin < telemetry.track_manager.tmax and images.tmax > telemetry.track_manager.tmin: # overlap
                        telemetries_for_images.append(telemetry)
                except KeyError:
                    print('Got KeyError when trying to read from', telemetry.fname, 'so skipping.')
        if len(telemetries_for_images) > 0:
            paired_recordings.append(PairedRecording(images, telemetries_for_images))
        images.unload_data()  # Free memory for now.

    data = {
        'npz': img_recs,
        'bin': tel_recs,
        'truncated_bin': trunc_tel_recs,
        'nontrunc_tel_recs': nontrunc_tel_recs,
        'paired': paired_recordings,
    }

    if save_known:
        save_known_pairings(data, base_dir)

    return data


def show_detections_on_image(img, t_image, track_managers, obstructed_color=(0, 255, 0), unobstructed_color=(0, 0, 255), offset=0):
    show_image = np.copy(img)
    all_active = []
    for track_manager in track_managers:
        active = track_manager.get_active_tracks(t_image)
        all_active.extend(active)
        for track in active:
            if t_image+offset > track.tmax or t_image+offset < track.tmin:
                continue
            screenx = track.get('screenx', t_image+offset)
            screeny = track.get('screeny', t_image+offset)
            occluded = track.get('is_occluded', t_image+offset)
            if screenx != -1 and screeny != -1:
                r = int(show_image.shape[0]*screeny)
                c = int(show_image.shape[1]*screenx)
                R = 8
                cv2.circle(show_image, (c, r), R, obstructed_color if occluded else unobstructed_color, 3)
    return show_image, all_active


def write_animation(
        itr: Iterator[np.array],
        out_file: Path,
        dpi: int = 50,
        fps: int = 30,
        title: str = "Animation",
        comment: Optional[str] = None,
        writer: str = "pillow",
    ) -> None:
    """Function that writes an animation from a stream of input tensors.

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        dpi: Dots per inch for output image.
        fps: Frames per second for the video.
        title: Title for the video metadata.
        comment: Comment for the video metadata.
        writer: The Matplotlib animation writer to use (if you use the
            default one, make sure you have `ffmpeg` installed on your
            system).
    """

    first_img = itr[0]
    height, width, _ = first_img.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))

    # Ensures that there's no extra space around the image.
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None,
    )

    # Creates the writer with the given metadata.
    Writer = mpl.animation.writers[writer]
    metadata = {
        "title": title,
        "artist": __name__,
        "comment": comment,
    }
    mpl_writer = Writer(
        fps=fps,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    with mpl_writer.saving(fig, out_file, dpi=dpi):
        im = ax.imshow(first_img, interpolation="nearest")
        ax.grid(False)
        mpl_writer.grab_frame()

        for img in tqdm(itr[1:], unit='images', desc='Writing video'):
            im.set_data(img)
            mpl_writer.grab_frame()
    

if __name__ == '__main__':
    
    
    # import sys.path
    # sys.path.append(join(HERE, '..'))
    data = find_filenames()
    for paired_recording in tqdm(data['paired']):
        paired_recording.write_video()
        paired_recording.unload_image_data()
        
