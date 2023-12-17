
import pickle

import pandas as pd
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from consts import JULY
from models.global_data import get_global_data

with open('playground_input.pkl', 'rb') as f:
    data = pickle.load(f)

X, val_X, test_X, y, val_y = get_global_data()

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Create sequence from rows 0 to idx
        sequence = self.X.iloc[:(idx + 1)].values
        label = self.y.iloc[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
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
        return out[:, -1, :]  # Return the outputs for the last time step

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
    X = X.drop(columns=['date', 'forecast_year']).reset_index(drop=True)
    if train:
        return SequenceDataset(X, y)

    raise NotImplementedError



if __name__ == '__main__':

    lr = 1e-3
    bs = 128
    X = X[X.date.dt.month <= JULY]
    train_set = features2seqs(X, y) # todo see we can overfit to a small training set before continuing
    combined_X = pd.concat([X, val_X])
    combined_y = pd.concat([y, val_y])
    combined_set = features2seqs(combined_X, combined_y)

    dataloader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=pad_collate_fn)

    n_feats = train_set[0][0].shape[1]
    model = LSTMModel(input_size=n_feats)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # todo implement AQM loss, requires multioutput (use dummy std for starters)
    num_epochs = 50
    for epoch in range(num_epochs):
        for sequences, labels, lengths in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            outputs = outputs.squeeze() # todo remove/change when using a multioutput
            # Ensure labels are also squeezed to match output shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#%%
