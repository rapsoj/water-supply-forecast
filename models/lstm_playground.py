import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from consts import JULY, DEF_QUANTILES
from fitters.lstm_utils import train_lstm


@dataclass
class HypParams:
    lr: float
    bs: float
    n_epochs: int
    n_hidden: int
    hidden_size: int
    dropout_prob: float


DEF_LSTM_HYPPARAMS = HypParams(lr=1e-3, bs=8, n_epochs=10, n_hidden=1, hidden_size=64, dropout_prob=0.3)


def main():
    with open('global_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # X, y = data['train']
    # val_X, val_y = data['val']

    X, val_X, test_X, y, val_y = data
    y_mean, y_std = y.volume.mean(), y.volume.std()
    y.volume = (y.volume - y_mean) / y_std
    val_y.volume = (val_y.volume - y_mean) / y_std

    y_vol = y.volume

    train_emp_aqm = np.mean([mean_pinball_loss(y_vol, [y_vol.quantile(q)] * len(y), alpha=q) for q in DEF_QUANTILES])
    val_emp_aqm = np.mean([mean_pinball_loss(val_y.volume, [y_vol.quantile(q)] * len(val_y), alpha=q)
                           for q in DEF_QUANTILES])
    print(f"Empirical quantile's training loss: {train_emp_aqm:.3f}")
    print(f"Empirical quantile's validation loss: {val_emp_aqm:.3f}")

    train_set = features2seqs(X, y)

    val_set = features2seqs(val_X, val_y)

    n_feats = train_set[0][0].shape[1]

    N_HIDDEN = [1, 2, 3]
    DROPOUT_PROBS = [0.2, 0.3, 0.4, 0.5]
    HIDDEN_SIZES = [16, 32, 64, 128, 256]
    BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
    LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    N_EPOCHS = [5, 10, 15, 20, 25, 30]
    # todo implement hyperparam search

    dataloader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=pad_collate_fn)
    model = LSTMModel(input_size=n_feats)

    train_lstm(dataloader, val_set, model, lr)


if __name__ == '__main__':
    main()
