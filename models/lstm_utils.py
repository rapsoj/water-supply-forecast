from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from consts import JULY, DEF_QUANTILES


@dataclass
class HypParams:
    lr: float
    bs: int
    n_epochs: int
    n_hidden: int
    hidden_size: int
    dropout_prob: float


DEF_LSTM_HYPPARAMS = HypParams(lr=1e-4, bs=2, n_epochs=7, n_hidden=1, hidden_size=512, dropout_prob=0.5)

class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        if self.y is not None:
            assert len(self.X) == len(self.y), 'Error - X and y have different lengths!'
        return len(self.X)

    def __getitem__(self, idx):
        idx_row = self.X.iloc[idx]
        fy = idx_row.forecast_year
        site_id = idx_row.site_id
        df_idx = idx_row.name
        X = self.X[self.X.site_id == site_id]

        if self.y is not None:
            label = self.y.iloc[idx]
            assert label.site_id == site_id, 'Error - site id mismatch!'

        assert np.all(X.forecast_year.iloc[:-1].values <= X.forecast_year.iloc[1:].values), \
            'Error - not sorted by forecast year!'

        init_ind = (X.forecast_year == fy).idxmax()
        # Create sequence from start of year until now
        sequence = X.loc[init_ind:df_idx].drop(columns=['forecast_year', 'site_id']).values
        if self.y is None:
            return torch.tensor(sequence, dtype=torch.float32)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label.volume, dtype=torch.float32)


def pad_collate_fn(batch):
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    batch_has_label = isinstance(batch[0], tuple)
    if batch_has_label:
        sequences, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        sequences = batch
    padded_sequences = pad_sequence(sequences, batch_first=True)
    lengths = [len(seq) for seq in sequences]
    if batch_has_label:
        return padded_sequences, labels, lengths
    return padded_sequences, lengths


def features2seqs(X: pd.DataFrame, y: pd.DataFrame = None):
    X = X[X.date.dt.month <= JULY].drop(columns=['date']).reset_index(drop=True)
    X.sort_values(by=['site_id', 'forecast_year'], inplace=True)
    if y is not None:
        y = y.iloc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    if y is not None:
        assert (X.site_id == y.site_id).all(), 'Error - site id mismatch!'
        assert (X.forecast_year == y.forecast_year).all(), 'Error - forecast year mismatch!'

    return SequenceDataset(X, y)


def quantile_loss(y_true, y_pred, quantile: float):
    return torch.mean(torch.max(quantile * (y_true - y_pred), -(1 - quantile) * (y_true - y_pred)))


def avg_quantile_loss(pred_means, pred_stds, y_true):
    losses = [quantile_loss(y_true, pred_means + norm.ppf(q) * pred_stds, q) for q in DEF_QUANTILES]
    return torch.mean(torch.stack(losses))


def calc_val_loss(model: nn.Module, val_set):
    if val_set is None:
        return

    model.eval()
    with torch.inference_mode():
        dataloader = DataLoader(val_set, collate_fn=pad_collate_fn)
        val_losses = []
        for sequences, labels, lengths in dataloader:
            means, stds = model(sequences, lengths)
            loss = avg_quantile_loss(means, stds, labels)
            val_losses.append(loss)
        model.train()
        return np.mean(val_losses)


def train_lstm(train_dloader: DataLoader, val_set: Dataset, model: nn.Module, lr: float, num_epochs: int) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = 0
        for sequences, labels, lengths in train_dloader:
            optimizer.zero_grad()
            means, stds = model(sequences, lengths)
            # Ensure labels are also squeezed to match output shape
            loss = avg_quantile_loss(means, stds, labels)
            assert loss.item() > 0, 'Error - loss is negative!'
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(sequences)

        train_loss /= len(train_dloader.dataset)
        val_loss = calc_val_loss(model, val_set)

        epoch_str = f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}'
        if val_loss is not None:
            epoch_str += f', Val Loss: {val_loss.item():.4f}'
        print(epoch_str)

    return model
