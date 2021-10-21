from numpy.core.numeric import full
import pytorch_lightning as pl
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class LossRecorder(pl.Callback):

    def __init__(self):
        self.losses = []
        self.val_losses = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
        if 'loss' in outputs:
            self.losses.append((
                len(self.losses)+1,
                # batch_idx,
                float(outputs['loss'])
            ))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if 'val_loss' in outputs:
            self.val_losses.append((
                len(self.losses), 
                # batch_idx,
                float(outputs['val_loss'])
            ))

    def show(self, save_path=None):
        fig, ax = plt.subplots(1, 1)
        B = [b for (b, l) in self.losses]
        L = [l for (b, l) in self.losses]
        ax.plot(B, L, label='Loss', color='black', alpha=0.5)

        if len(self.val_losses) > 0:
            B = [b for (b, l) in self.val_losses]
            L = [l for (b, l) in self.val_losses]
            ax.scatter(B, L, label='Validation Loss', color='blue', zorder=99, alpha=.8)

        ax.legend()
        ax.set_xlabel('Number of training batches seen')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        if save_path is not None:
            from os.path import join
            fig.savefig(save_path)

        return fig, ax


class PbarCallback(pl.Callback):

    def __init__(self, n_epochs, unit='epoch', desc='Training'):
        self.n_epochs = n_epochs
        self.unit = unit
        self.desc = desc

    @property
    def pbar(self):
        # Create this lazily so that all the crap that pytorch prints before training
        # is not printed over the progress bar.
        if not hasattr(self, '_pbar'):
            self._pbar = tqdm(total=self.n_epochs, unit=self.unit, desc=self.desc)
        return self._pbar

    def on_train_epoch_end(self, trainer, pl_module):
        self.pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.pbar.close()


def get_dataloaders_kwargs_from_arrays(input_array, output_array, batch_size=300, train_frac=.9, 
    input_dtype_assumed='float32',
    output_dtype='float32',
    num_workers=8,
    ):
    """
    Args:
        input_array: Numpy array of shape (n_samples, ...)
        output_array: Numpy array of shape (n_samples, ...)
        batch_size: Batch size
        train_frac: Fraction of data to use for training

    Returns:
        Dict of dataloader kwargs, suitable for passing to trainer.fit(
            model, **get_dataloaders_kwargs_from_arrays(...)[0]
        )
    """
    # # Split into train and test
    # n_samples = input_array.shape[0]
    # n_train = int(n_samples * train_frac)
    # train_input = input_array[:n_train]
    # train_output = output_array[:n_train]
    # test_input = input_array[n_train:]
    # test_output = output_array[n_train:]

    # # Create dataloaders
    # train_dataloader_kwargs = {
    #     'batch_size': batch_size,
    #     'shuffle': True,
    #     'num_workers': 0,
    #     'pin_memory': False,
    #     'drop_last': True,
    #     'collate_fn': lambda x: x,
    # }
    # train_dataloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(
    #         torch.from_numpy(train_input),
    #         torch.from_numpy(train_output)
    #     ),
    #     **train_dataloader_kwargs
    # )
    # test_dataloader_kwargs = {
    #     'batch_size': batch_size,
    #     'shuffle': False,
    #     'num_workers': 0,
    #     'pin_memory': False,
    #     'drop_last': False,
    #     'collate_fn': lambda x: x,
    # }
    # test_dataloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(
    #         torch.from_numpy(test_input),
    #         torch.from_numpy(test_output)
    #     ),
    #     **test_dataloader_kwargs

    full_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(input_array),
        torch.from_numpy(output_array.astype(output_dtype))
    )

    total_count = len(full_dataset)

    assert train_frac <= 1.0 and train_frac > 0.0, 'train_frac must be in (0, 1]'
    train_count = int(train_frac * total_count)
    val_count = total_count - train_count
    train_dataset, valid_dataset = torch.utils.data.random_split(
        full_dataset, (train_count, val_count)
    )

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    kwargs = dict(
        train_dataloaders=[train_dataset_loader],
        val_dataloaders=(None if not val_count else [torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )]),
    )

    return kwargs, full_dataset