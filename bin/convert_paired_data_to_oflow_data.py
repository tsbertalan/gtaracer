from posixpath import basename
import numpy as np
from tqdm.auto import tqdm
import cv2
from os.path import join, exists, dirname, abspath

import sys
HERE = dirname(abspath(__file__))
sys.path.append(join(HERE, '..', 'src'))
import gta
import gta.recording_videos
import gta.train_velocity_predictor



def main(base_dir=gta.default_configs.PROTOCOL_V2_DIR):

    data = gta.recording_videos.find_filenames(base_dir=base_dir)

    existing = [basename(f) for f in gta.train_velocity_predictor.list_existing_oflow_savefiles(base_dir)]

    for pairing in tqdm(data['paired'], unit='pairing'):

        npz_filename = pairing.save_name_base + gta.train_velocity_predictor.OFLOW_SAVE_SUFFIX

        if not exists(npz_filename):
            flow_data, vel_data, times, vvecs = pairing_to_arrays(pairing)

            np.savez(
                npz_filename,
                flow_data=flow_data,
                vel_data=vel_data,
                times=times,
                vvecs=vvecs,
            )

        else:
            assert basename(npz_filename) in existing


def pairing_to_arrays(pairing):
    telemetry_recording = pairing.telemetry_recordings[0]
    telemetry_recording.track_manager.merge_player_tracks()
    player_tracks = [track for track in telemetry_recording.track_manager.tracks 
                    if track.is_player and track.is_vehicle]

    player_tracks.sort(key=lambda t: t.duration)
    longest_player_track = player_tracks[-1]

    import gta.train_velocity_predictor

    times, images, velocities, directional_velocities, vvecs, poses, forward_vectors, flow_from_previous = \
        gta.train_velocity_predictor.pair_images_with_ego_velocities(pairing, LIMIT=None)

    new_shape = 400, 300

    def shrink_img(img):
        return cv2.resize(img, new_shape)

    flow_data = np.stack([
            shrink_img(im)
            for im in flow_from_previous[1:]
        ], axis=0).astype('float32')

    vel_data = np.array(directional_velocities[1:]).reshape((-1, 1)).astype('float32')

    flow_data_NCHW = np.rollaxis(flow_data, -1, 1)

    return flow_data_NCHW, vel_data, times, vvecs


if __name__ == '__main__':
    main()
