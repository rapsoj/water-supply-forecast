import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_pinball_loss
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from consts import JULY, DEF_QUANTILES


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        assert np.all(self.X.forecast_year.iloc[:-1].values <= self.X.forecast_year.iloc[1:].values), \
            'Error - not sorted by forecast year!'

        fy = self.X.forecast_year.iloc[idx]
        init_ind = (self.X.forecast_year == fy).argmax()
        # Create sequence from rows 0 to idx
        sequence = self.X.iloc[init_ind:idx + 1].drop(columns='forecast_year').values
        label = self.y.iloc[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout_prob: float = 0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        x_packed = pack_padded_sequence(x, lengths, batch_first=True)
        out_packed, _ = self.lstm(x_packed)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
        # Apply the linear layer to the unpacked outputs
        out = self.fc(out_padded)
        return out[torch.arange(out.size(0)), np.array(lengths) - 1]  # Return the outputs for the last time step


def pad_collate_fn(batch):
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    # Pad the sequences and stack the labels
    padded_sequences = pad_sequence(sequences, batch_first=True)
    lengths = [len(seq) for seq in sequences]
    labels = torch.stack(labels)
    return padded_sequences, labels, lengths


def features2seqs(X: pd.DataFrame, y: pd.Series, train: bool = True):
    X = X[X.date.dt.month <= JULY].drop(columns=['date'])
    if train:
        return SequenceDataset(X.iloc[:-1], y.iloc[:-1])

    raise NotImplementedError


def quantile_loss(quantile: float):
    def qloss(y_true, y_pred):
        return torch.mean(torch.max(quantile * (y_true - y_pred), -(1 - quantile) * (y_true - y_pred)))

    return qloss


def avg_quantile_loss(y_pred, y_true):
    return torch.mean(torch.stack([quantile_loss(q)(y_true, y_pred) for q in DEF_QUANTILES]))


def calc_val_loss(model: nn.Module, val_set):
    with torch.inference_mode():
        dataloader = DataLoader(val_set, collate_fn=pad_collate_fn)
        val_losses = []
        for sequences, labels, lengths in dataloader:
            outputs = model(sequences, lengths)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            val_losses.append(avg_quantile_loss(outputs, labels))
        return np.mean(val_losses)


def train_lstm(train_dloader: DataLoader, val_set: Dataset, model: nn.Module, lr: float):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = avg_quantile_loss  # todo implement AQM loss, requires multioutput (use dummy std for starters)

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = 0
        for sequences, labels, lengths in train_dloader:
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            outputs = outputs.squeeze()  # todo remove/change when using a multioutput
            # Ensure labels are also squeezed to match output shape
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(sequences)

        train_loss /= len(train_dloader.dataset)
        val_loss = calc_val_loss(model, val_set)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')


def main():
    with open('playground_input.pkl', 'rb') as f:
        data = pickle.load(f)

    X, y = data['train']
    val_X, val_y = data['val']

    train_set = features2seqs(X, y)

    val_set = features2seqs(val_X, val_y)

    bs = 8
    lr = 1e-3

    dataloader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=pad_collate_fn)

    n_feats = train_set[0][0].shape[1]
    model = LSTMModel(input_size=n_feats)

    train_lstm(dataloader, val_set, model, lr)


if __name__ == '__main__':
    main()
