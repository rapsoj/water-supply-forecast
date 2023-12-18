import os
import pickle
from itertools import product
from random import shuffle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from torch.utils.data import DataLoader

from consts import DEF_QUANTILES
from models.fitters import LSTMModel
from models.lstm_utils import train_lstm, pad_collate_fn, features2seqs, calc_val_loss, HypParams, DEF_LSTM_HYPPARAMS


def main():
    with open(os.path.join('..', 'assets', 'data', 'global_data_for_lstm_debugging.pkl'), 'rb') as f:
        data = pickle.load(f)

    # X, y = data['train']
    # val_X, val_y = data['val']

    # todo use regular standardisation, less copying
    X, val_X, test_X, y, val_y = data
    y_mean, y_std = y.volume.mean(), y.volume.std()
    y.volume = (y.volume - y_mean) / y_std
    val_y.volume = (val_y.volume - y_mean) / y_std

    train_emp_aqm = y.groupby('site_id').volume \
        .apply(lambda x: np.mean([mean_pinball_loss(x, [x.quantile(q)] * len(x), alpha=q) for q in DEF_QUANTILES])) \
        .mean()
    val_emp_aqm = y.groupby('site_id').volume \
        .apply(lambda x: np.mean([mean_pinball_loss(x, [y.volume[y.site_id == x.name].quantile(q)] * len(x), alpha=q)
                                  for q in DEF_QUANTILES])) \
        .mean()
    print(f"Sitewise empirical quantile's training loss: {train_emp_aqm:.3f}")
    print(f"Sitewise empirical quantile's validation loss: {val_emp_aqm:.3f}")

    train_set = features2seqs(X, y)
    val_set = features2seqs(val_X, val_y)

    n_feats = train_set[0][0].shape[1]

    N_HIDDEN = [1, 2, 3]
    DROPOUT_PROBS = [0.2, 0.3, 0.4, 0.5]
    HIDDEN_SIZES = [32, 64, 128, 256, 512]
    BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
    LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    N_EPOCHS = list(range(4, 15, 3))

    # hypparam search
    hyp_params_combs = list(product(LEARNING_RATES, BATCH_SIZES, N_EPOCHS, N_HIDDEN, HIDDEN_SIZES, DROPOUT_PROBS))
    hyp_params_combs = [HypParams(*hyp_params) for hyp_params in hyp_params_combs]
    shuffle(hyp_params_combs)

    results = []
    for hyp_params in [DEF_LSTM_HYPPARAMS]:
        if hyp_params.n_hidden == 1 and hyp_params.dropout_prob != min(DROPOUT_PROBS):
            continue  # arbitrary dropout prob, prevent recurring computations

        dataloader = DataLoader(train_set, batch_size=hyp_params.bs, shuffle=True, collate_fn=pad_collate_fn)
        model = LSTMModel(input_size=n_feats, hidden_size=hyp_params.hidden_size, num_layers=hyp_params.n_hidden,
                          dropout_prob=hyp_params.dropout_prob)

        model = train_lstm(dataloader, val_set, model, lr=hyp_params.lr, num_epochs=hyp_params.n_epochs)

        val_loss = calc_val_loss(model, val_set)
        results.append({'val_loss': val_loss} | hyp_params.__dict__)

        res_str = f'val loss: {val_loss:.5f}, {hyp_params}\n'
        # append new results to running text file
        # with open('lstm_results.txt', 'a+') as f:
        #     f.write(res_str)

        print(res_str)

    # pd.DataFrame.from_records(results).to_csv('lstm_results.csv')


if __name__ == '__main__':
    main()
