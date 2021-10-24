from os.path import join, dirname, abspath, basename, exists
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm    
import pytorch_lightning as pl

import sys
HERE = dirname(abspath(__file__))
sys.path.append(join(HERE, '..', 'src'))

import gta.train_velocity_predictor
import gta.neural


def show_velocity_courses(base_dir=gta.default_configs.PROTOCOL_V2_DIR):
    model_save_dir = join(base_dir, 'models')
    
    oflow_data_paths = gta.train_velocity_predictor.list_existing_oflow_savefiles(base_dir)

    for path in tqdm(oflow_data_paths):

        data = np.load(path)

        try:
            outp = data['vel_data']

        except EOFError:
            print('EOFError:', path)
            continue

        fig, ax = plt.subplots()
        ax.plot(outp.ravel())
        ax.set_ylim(0, 60)
        fig.savefig(join(model_save_dir, basename(path)+'-vel_predictions.png'));
        
    plt.show()


def main(
    base_dir=gta.default_configs.PROTOCOL_V2_DIR, 
    n_epochs=4000,  batch_size=200,
    LIMIT_OFLOWFILES=None, START_OFLOWFILES=0,
    # LIMIT_OFLOWFILES=1, START_OFLOWFILES=0,
    # n_epochs=256, LIMIT_OFLOWFILES=1, batch_size=128,
    n_gpus=1, n_dataloader_workers=8,
    reload_from_previous=True,
    ):
    model_save_dir = join(base_dir, 'models')
    
    oflow_data_paths = gta.train_velocity_predictor.list_existing_oflow_savefiles(base_dir)

    output_arrays = []

    aux_data_per_path = {}

    session_lengths = []

    if LIMIT_OFLOWFILES is None:
        LIMIT_OFLOWFILES = len(oflow_data_paths)
    if START_OFLOWFILES is None:
        START_OFLOWFILES = 0

    paths = oflow_data_paths[START_OFLOWFILES:]
    print('Trying to load', len(paths), 'oflow npz files.')
    from tempfile import mkdtemp
    input_memmap_path = join(mkdtemp(), 'input_array.npy')
    print('Memmap will be saved in', input_memmap_path)
    input_mmap_initialized = False
    
    for path in tqdm(paths, unit='path', desc='Loading data into memmap'):

        data = np.load(path)

        try:
            inp = data['flow_data']
            outp = data['vel_data']

        except EOFError:
            print('EOFError:', path)
            continue

        session_lengths.append(inp.shape[0])
        if not input_mmap_initialized:
            input_mmap = np.memmap(input_memmap_path, dtype='float32', mode='w+', shape=inp.shape)
            input_mmap_initialized = True
        else:
            input_mmap.flush()
            new_shape = list(input_mmap.shape)
            new_shape[0] += inp.shape[0]
            input_mmap = np.memmap(input_memmap_path, dtype='float32', mode='r+', shape=tuple(new_shape))
        input_mmap[-inp.shape[0]:] = inp
        input_mmap.flush()

        output_arrays.append(outp)
        aux_data_per_path[path] = {}
        if 'times' in data:
            aux_data_per_path[path]['times'] = data['times']
        if 'vvecs' in data:
            aux_data_per_path[path]['vvecs'] = data['vvecs']

        if len(output_arrays) >= LIMIT_OFLOWFILES:
            break
        

    print('Successfully loaded', len(output_arrays), 'oflow files.')

    output_array = np.concatenate(output_arrays, axis=0)

    print('input_array.shape:', input_mmap.shape)
    print('output_array.shape:', output_array.shape)

    dataloaders, _unused_full_dataset = gta.neural.get_dataloaders_kwargs_from_arrays(
        input_mmap, output_array, 
        num_workers=n_dataloader_workers, batch_size=batch_size,
    )

    print('Training with', len(dataloaders['train_dataloaders'][0].dataset), 'samples.')
    print('batch_size:', batch_size)
    batches_per_epoch = len(dataloaders['train_dataloaders'][0])
    print('batches_per_epoch maybe:', batches_per_epoch)
    print('Validating with', len(dataloaders['val_dataloaders'][0].dataset), 'samples.')


    reloaded = False
    save_path = gta.default_configs.OFLOW_VEL_MODEL_SAVE_PATH
    if reload_from_previous:
        if not exists(save_path):
            print('No previous model found at', save_path)
        else:
            # Load the model.
            print('Loading from', save_path)
            model = gta.train_velocity_predictor.VelocityPredictorFromOpticalFlow.load_from_checkpoint(save_path)
            reloaded = True

    if not reloaded:
        print('Creating new model from scratch.')
        model = gta.train_velocity_predictor.VelocityPredictorFromOpticalFlow(
            epochs_for_scheduler=n_epochs, 
            batches_per_epoch_for_scheduler=batches_per_epoch,
        )

    # Make a minimal batch of predictions so that the parameters are initialized.
    model.predict_from_numpy(input_mmap[[0]])

    loss_recorder = gta.neural.LossRecorder()

    class EveryKEpochsPlot(pl.Callback):

        def __init__(self, k, loss_recorder):
            self.k = k
            self.loss_recorder = loss_recorder

        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch % self.k == 0:
                viz_fig, ax = show_status(model, input_mmap, output_array, session_lengths)

                for save_path in [
                    join(model_save_dir, 'velvid', 'vel_predictions-%05d_epochs.png' % trainer.current_epoch),
                    join(HERE, 'vel_predictions.png'),
                ]:
                    viz_fig.savefig(save_path)

                loss_fig, loss_ax = self.loss_recorder.show(save_path=None)
                for save_path in [
                    join(
                        model_save_dir, 'lvid', 'loss-%05d_epochs.png' % trainer.current_epoch
                    ),
                    join(HERE, 'loss.png'),
                ]:
                    loss_fig.savefig(save_path)


                return viz_fig, loss_fig

    trainer = pl.Trainer(
        max_epochs=n_epochs, min_epochs=int(np.math.ceil(n_epochs*.025)), weights_save_path=model_save_dir, 
        progress_bar_refresh_rate=0,
        gpus=n_gpus,
        callbacks=[
            loss_recorder,
            gta.neural.PbarCallback(n_epochs),
            # pl.callbacks.EarlyStopping(patience=120, monitor='val_loss'),  # Needs a validation set.
            EveryKEpochsPlot(1, loss_recorder),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        log_every_n_steps=50,
        logger=pl.loggers.TensorBoardLogger(
            join(model_save_dir, 'tb_logs'), 
            name='oflow_vel',
            default_hp_metric=False,
        ),
    )
    trainer.fit(model, **dataloaders)

    loss_recorder.show(join(model_save_dir, 'loss.png'))

    # Save the model.
    print('Saving to', save_path)
    trainer.save_checkpoint(save_path)


def show_status(model, input_array, output_array, session_lengths):

    # Close all matplotlib figures
    plt.close('all')

    fig, ax = plt.subplots()
    n_plot = 10000

    all_indices = np.arange(len(input_array))
    indices_per_recording = []
    i1 = 0
    for di in session_lengths:
        indices_per_recording.append(all_indices[i1:i1+di])
        i1 += di
    pred_indices = indices_per_recording[min(3, len(indices_per_recording)-1)][:n_plot]
    predictions = model.predict_from_numpy(input_array[pred_indices])
    approx_dt = .1
    times = np.arange(len(pred_indices)+1).astype(float) * approx_dt
    ax.plot(times[1:] - times[1], output_array[pred_indices], label='Truth')
    ax.plot(times[1:] - times[1], predictions.ravel(), label='Predictions')
    ax.set_ylabel('Time $[s]$')
    ax.legend()
    ax.set_ylabel('Signed Velocity $[m/s]$')
    ax.set_xlim(0, 20);
    fig.tight_layout()
    
    return fig, ax


if __name__ == '__main__':
    main()
    # show_velocity_courses()
