import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_pinball_loss
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from consts import JULY, DEF_QUANTILES

with open('playground_input.pkl', 'rb') as f:
    data = pickle.load(f)

X, y = data['train']
val_X, val_y = data['val']


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
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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


bs = 8
lr = 1e-3


def quantile_loss(quantile: float):
    def qloss(y_true, y_pred):
        return torch.mean(torch.max(quantile * (y_true - y_pred), -(1 - quantile) * (y_true - y_pred)))

    return qloss


def avg_quantile_loss(y_pred, y_true):
    return torch.mean(torch.stack([quantile_loss(q)(y_true, y_pred) for q in DEF_QUANTILES]))


train_set = features2seqs(X, y)  # todo see we can overfit to a small training set before continuing
combined_X = pd.concat([X, val_X])
combined_y = pd.concat([y, val_y])
combined_set = features2seqs(combined_X, combined_y)

dataloader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=pad_collate_fn)

n_feats = train_set[0][0].shape[1]
model = LSTMModel(input_size=n_feats)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = avg_quantile_loss  # todo implement AQM loss, requires multioutput (use dummy std for starters)

num_epochs = 100
for epoch in range(num_epochs):
    for sequences, labels, lengths in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        outputs = outputs.squeeze()  # todo remove/change when using a multioutput
        # Ensure labels are also squeezed to match output shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

np.mean([mean_pinball_loss(y, [y.quantile(q)] * len(y), alpha=q) for q in DEF_QUANTILES])
