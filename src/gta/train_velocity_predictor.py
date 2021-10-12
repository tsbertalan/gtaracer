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




class VelocityPredictor(pl.LightningModule):
    def __init__(self, window_size=4, color_channels=3):
        super().__init__()

        in_channels = color_channels * window_size

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels, 4, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4, 6, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 4, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.LazyLinear(4),
            nn.ReLU(),
            
            nn.Linear(4, 2),
            nn.ReLU(),

            nn.Linear(2, 1),

        )

    def forward(self, images_window):
        return self.layers(images_window)

    def training_step(self, batch, batch_idx):
        images_window, velocities = batch[0]
        predictions = self(images_window)
        loss = F.mse_loss(predictions, velocities)
        return {'loss': loss}

    def configure_optimizers(self, lr=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr)



def get_windowed_data():
    # Create windowed data
    data_dir = gta.default_configs.VELOCITY_DATA_DIR
    from glob import glob

    all_data = {
            'train_windows': [], 
            'window_velocities': [],
            'window_times': [],
            'npz_data': [],
        }

    query = join(data_dir, '*.reduced.npz')
    paths = glob(query)
    print('Query', query, 'yielded', len(paths), 'files.')
    for data_path in tqdm(paths, unit='file', desc='load data'):

        data = np.load(data_path)

        window_size = 4

        target_img_height = 32

        n, h, w, c = data['images'].shape
        aspect_ratio = w / h
        target_img_width = int(target_img_height * aspect_ratio)
        
        scaled_images = np.stack([
            skimage.transform.resize(img, (target_img_height, target_img_width))
            for img in data['images']
        ]).astype('single')

        # Change data from n, h, w, c to n, c, h, w
        scaled_images = np.transpose(scaled_images, (0, 3, 1, 2))

        velocities = data['v']

        overlapping_windows = [
            np.concatenate(scaled_images[i:i+window_size], axis=0)
            for i in range(0, len(scaled_images) - window_size + 1)
        ]
        window_velocities = velocities[window_size-1:].astype('single').reshape((-1, 1))

        train_windows = np.stack(overlapping_windows)

        all_data['train_windows'].append(train_windows)
        all_data['window_velocities'].append(window_velocities)
        all_data['npz_data'].append(data)
        all_data['window_times'].append(data['t'][window_size-1:].reshape((-1, 1)))
        assert len(all_data['train_windows'][-1]) == len(all_data['window_velocities'][-1])
        assert len(all_data['train_windows'][-1]) == len(all_data['window_times'][-1])
        
    return all_data


def get_model_save_dir():
    return join(HERE, 'models')





def train(batch_size=32):
    pl.seed_everything(42)

    model_save_dir = get_model_save_dir()    

    from sys import path
    path.append(join(HERE, '..', 'src'))
    from gta.utils import mkdir_p
    mkdir_p(model_save_dir)

    data = get_windowed_data()
    train_windows = np.concatenate(data['train_windows'], axis=0)
    window_velocities = np.concatenate(data['window_velocities'], axis=0)

    window_size = train_windows.shape[1] // 3
    print(window_size)


    print('Predict data with shape', window_velocities.shape, 'from data with shape', train_windows.shape, '.')

    model = VelocityPredictor(window_size=window_size)

    # Create PyTorch datasets from the data.
    full_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_windows), torch.from_numpy(window_velocities))
    

    # Create a DataLoader for the training dataset.
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)


    # window_loader = DataLoader(train_windows, batch_size=batch_size, shuffle=True)
    # velocity_loader = DataLoader(window_velocities, batch_size=batch_size, shuffle=True)

    class LossRecorder(pl.Callback):

        def __init__(self):
            self.losses = []

        def on_before_backward(self, trainer, pl_module, loss):
            # # Do garbage collection.
            # import gc
            # gc.collect()
            self.losses.append(float(loss))

    class PbarCallback(pl.Callback):

        def __init__(self, n_epochs):
            self.pbar = tqdm(total=n_epochs, unit='epoch', desc='Training')

        def on_epoch_end(self, trainer, pl_module):
            self.pbar.update(1)

        def on_train_end(self, trainer, pl_module):
            self.pbar.close()

    loss_recorder = LossRecorder()

    n_epochs = 10240
    trainer = pl.Trainer(
        max_epochs=n_epochs, min_epochs=n_epochs, weights_save_path=model_save_dir, 
        progress_bar_refresh_rate=0,
        callbacks=[
            loss_recorder,
            PbarCallback(n_epochs),
            # pl.callbacks.EarlyStopping(patience=4, monitor='loss'),  # Needs a validation set.
        ]
    )
    trainer.fit(model, train_dataloaders=[train_loader])

    fig, ax = plt.subplots(1, 1)
    l = loss_recorder.losses
    ax.plot(l, label='history loss')
    ax.legend()
    ax.set_xlabel('batch')
    ax.set_ylabel('loss')
    ax.set_yscale('log')
    fig.savefig(join(model_save_dir, 'loss.png'))


def reload():
    model_save_dir = get_model_save_dir()    

    
    data = get_windowed_data()
    train_windows = np.concatenate(data['train_windows'], axis=0)
    window_velocities = np.concatenate(data['window_velocities'], axis=0)
    window_times = np.concatenate([
        a - a.min() for a in data['window_times']
    ], axis=0)
    
    window_size = train_windows.shape[1] // 3
    print(window_size)

    model = VelocityPredictor(window_size=window_size)
    from subprocess import check_output
    print('Looking for model in', model_save_dir, '...')
    checkpoints = check_output(['find', model_save_dir, '-iname', '*.ckpt']).decode('utf-8').split('\n')
    checkpoints = list(sorted(checkpoints))
    print('Among\n    ', '\n    '.join(checkpoints))
    ckpt = checkpoints[-1]
    print('Selected checkpoint', ckpt)
    loaded = torch.load(ckpt)

    model.load_state_dict(loaded['state_dict'])

    predictions = model(torch.from_numpy(train_windows)).detach().numpy()


    for i_pred in np.linspace(0, len(train_windows)-1, 10).astype('int'):
        first_window = train_windows[i_pred]
        
        fig, axes = plt.subplots(ncols=window_size, figsize=(window_size * 3, 3))
        for i in range(window_size):
            ax = axes[i]
            im = first_window[i*3:(i+1)*3]
            # Convert back to h, w, c
            im = np.transpose(im, (1, 2, 0))
            ax.imshow(im)
            ax.set_title('$t=%.2f$' % window_times[i])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('datum {i}: predicted velocity {pred:.2f}, vs actual {actual:.2f}'.format(
            i=i_pred,
            pred=float(predictions[i_pred]*3600),
            actual=float(window_velocities[i_pred]*3600),
        ))
        fig.tight_layout()
        fig.savefig(join(model_save_dir, 'prediction_%02d.png' % i_pred))



    fig, ax = plt.subplots()
    ax.plot(window_velocities, label='true')
    ax.plot(predictions, label='predicted')
    ax.legend()
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Velocity [mi/h]')
    fig.tight_layout()
    fig.savefig(ckpt[:-5] + '.png')
    

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

        self.epochs_for_scheduler = epochs_for_scheduler
        self.batches_per_epoch_for_scheduler = batches_per_epoch_for_scheduler
        in_channels = flow_channels

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels, 4, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(4, 6, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 4, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.LazyLinear(4),
            nn.ReLU(),
            
            nn.Linear(4, 2),
            nn.ReLU(),

            nn.Linear(2, 1),

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

    def configure_optimizers(self, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-3, 
            epochs=self.epochs_for_scheduler, steps_per_epoch=self.batches_per_epoch_for_scheduler,
            verbose=False,
        )
        return [optimizer], [scheduler]


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