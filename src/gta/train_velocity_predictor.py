import numpy as np
from tqdm.auto import tqdm
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
from glob import glob
import cv2

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..'))
from gta.recording_videos import ImageRecording, TelemetryRecording

from os.path import join, dirname, basename

HERE = dirname(__file__)

import gta


def show_pairing_times(pairing, show_player_vehs=True):
    fig, ax = plt.subplots()
    labels = []
    Y = []
    X = []
    for i, item in enumerate([pairing.image_recording] + list(pairing.telemetry_recordings)):
        Y.append([i, i])
        
        if isinstance(item, ImageRecording):
            if len(item.times):
                t1, t2 = item.tmin, item.tmax
            else:
                t1, t2 = None, None
        else:
            t1, t2 = item.track_manager.tmin, item.track_manager.tmax
        X.append([t1, t2])
        bn = basename(item.fname)
        labels.append(bn)
        color = 'red' if isinstance(item, ImageRecording) else 'black'
        ax.plot(
            X[-1], Y[-1], linestyle='-', 
            color=color, linewidth=1,
        )
        if isinstance(item, TelemetryRecording) and show_player_vehs:
            player_vehs = [track for track in item.track_manager.tracks if track.is_player and track.is_vehicle]
            for pv in tqdm(player_vehs):
                if pv.duration < 1:
                    continue
                ax.plot(
                    [pv.tmin, pv.tmax], Y[-1],
                    color=color, linestyle='-',
                    alpha=0.25, linewidth=6, marker='|', markersize=10,
                )
    ax.set_yticks([y[0] for y in Y])
    ax.set_yticklabels(labels)
    ax.set_xlabel('wall_time')
    ax.set_title('Recordings paired with %s' % basename(pairing.image_recording.fname))


OFLOW_SAVE_SUFFIX = '_paired_data.npz'

def list_existing_oflow_savefiles(base_dir=gta.default_configs.PROTOCOL_V2_DIR):
    glob_pattern = join(base_dir, '*' + OFLOW_SAVE_SUFFIX)
    return glob(glob_pattern)


def shrink_img_for_oflow(img):
    new_shape = 400, 300
    return cv2.resize(img, new_shape)


def reshape_oflow_for_net(data):
    if len(data.shape) == 3:
        # Add a batch dimension.
        data = data[None, ...]
    
    if data.shape[-1] == 2:
        # Reshape to NCWH for PyTorch.
        data = data.transpose(0, 3, 1, 2)
    
    return data
        

def get_model_save_dir():
    return join(HERE, 'models')


def show_data():
    data = get_windowed_data()
    
    for i, (times, vel) in enumerate(zip(data['window_times'], data['window_velocities'])):
        fig, ax = plt.subplots()
        ax.plot(times.ravel(), vel.ravel(), marker='.')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [mi/h]')
        fig.tight_layout()
        save_path = join(get_model_save_dir(), 'velocity_%02d.png' % i)
        fig.savefig(save_path)
        print('Saved', save_path)


def convert_image_pair_to_optical_flow(img1, img2):

    # Convert to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #x_part = flow[:, :, 0]
    #y_part = flow[:, :, 1]

    return flow


class VelocityPredictorFromOpticalFlow(pl.LightningModule):
    def __init__(self, epochs_for_scheduler=100, batches_per_epoch_for_scheduler=10):
        flow_channels = 2
        super().__init__()

        self.start_lr = 1e-3
        self.max_lr = 1e-2

        self.epochs_for_scheduler = epochs_for_scheduler
        self.batches_per_epoch_for_scheduler = batches_per_epoch_for_scheduler
        in_channels = flow_channels

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels, 8, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 12, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.LazyLinear(24),
            nn.ReLU(),
            
            nn.Linear(24, 8),
            nn.ReLU(),

            nn.Linear(8, 1),

        )

    def forward(self, flow_images):
        return self.layers(flow_images)

    def predict_from_numpy(self, flow_images):
        return self.forward(torch.from_numpy(flow_images).to(self.device)).detach().cpu().numpy()

    def training_step(self, batch, batch_idx):
        flow_images, velocities = batch[0]  # For some reason, in the training step, we get a list (len=1) of lists (len=2) of tensors,
                                            # but in the validation step, we get a list (len=2) of tensors directly.
        predictions = self(flow_images)
        loss = F.mse_loss(predictions, velocities)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {
            'loss': loss,
            'log': {
                'train_loss': loss.detach(),
            },
        }

    def validation_step(self, batch, batch_idx):
        flow_images, velocities = batch
        predictions = self(flow_images)
        loss = F.mse_loss(predictions, velocities)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {
            'val_loss': loss,
            'log': {
                'val_loss': loss.detach(),
            },
        }

    def configure_optimizers(self, lr=None):
        if lr is None:
            lr = self.start_lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

import scipy.spatial.transform
def get_local_to_global_rot(st):
#     yaw = np.radians(state_tuple.yaw)
#     pitch = np.radians(state_tuple.pitch)
#     roll = np.radians(state_tuple.roll)
    return scipy.spatial.transform.Rotation.from_euler(
        seq='XZY', 
        angles=[st.pitch, st.yaw, st.roll], # Based on UVN camera from here http://athena.ecs.csus.edu/~gordonvs/165/resources/03-FundamentalsOf3D.pdf
        degrees=True
    )


def pair_images_with_ego_velocities(pairing, LIMIT=None):
    times = []
    images = []
    velocities = []
    directional_velocities = []
    forward_vectors = []
    vvecs = []
    poses = []
    flow_from_previous = []
    try:
        if LIMIT is None:
            LIMIT = len(pairing.image_recording.images)
        for iimg, (img, t) in enumerate(zip(tqdm(pairing.image_recording.images[:LIMIT], unit='frames'), pairing.image_recording.times[:LIMIT])):
            vel_recorded = False
            for telemetry_recording in pairing.telemetry_recordings:            
                tm = telemetry_recording.track_manager
                for track in tm.get_active_tracks(t):
                    if track.is_vehicle and track.is_player:
                        st = track.get_interpolated_state(t)
                        R = get_local_to_global_rot(st)
                        p = R.apply(np.array([0, 1, 0]))
                        v = np.array([st.velx, st.vely, st.velz])
#                         yaw = np.radians(st.yaw)
#                         pitch = np.radians(st.pitch)
#                         p = np.array([
#                             np.sin(yaw)*np.cos(pitch), 
#                             np.cos(yaw)*np.cos(pitch), 
#                             np.sin(pitch)
#                         ])
                        forward = (v @ p) > 0
                        vel_meters_per_second = np.linalg.norm(v) * (1 if forward else -1)
                        velocities.append(vel_meters_per_second)
                        directional_velocities.append(v @ p)
                        vvecs.append(v)
                        poses.append(np.array([st.posx, st.posy, st.posz, st.roll, st.pitch, st.yaw]))
                        forward_vectors.append(p)
                        vel_recorded = True
                        break
                if vel_recorded:
                    break
            if vel_recorded:
                images.append(img)
                times.append(t)
                if iimg > 0:
                    flow = convert_image_pair_to_optical_flow(images[iimg-1], images[iimg])
                    flow_from_previous.append(flow)
                else:
                    flow_from_previous.append(None)
    except KeyboardInterrupt:
        imax = min([len(z) for z in (times, images, velocities, directional_velocities, vvecs, poses, forward_vectors, flow_from_previous)])
        times = times[:imax]
        images = images[:imax]
        velocities = velocities[:imax]
        directional_velocities = directional_velocities[:imax]
        vvecs = vvecs[:imax]
        poses = poses[:imax]
        forward_vectors = forward_vectors[:imax]
        flow_from_previous = flow_from_previous[:imax]

    return times, images, velocities, directional_velocities, vvecs, poses, forward_vectors, flow_from_previous

if __name__ == '__main__':
    # train()
    # reload()
    show_data()