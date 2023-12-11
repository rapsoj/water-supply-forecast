import pandas as pd
from models.models import general_pcr_fitter, xgboost_fitter, k_nearest_neighbors_fitter
from enum import Enum


class Ensemble_Type(Enum):
    AVERAGE = 0
    BEST_PREDICTION = 1


def gen_basin_preds(train_site: pd.DataFrame, train_gt: pd.DataFrame, val_site: pd.DataFrame, val_gt: pd.DataFrame,
                    test: pd.DataFrame,
                    fitter=xgboost_fitter) -> tuple:  # todo reimplement ensembling so it happens elsewhere, so you can still store the individual model predictions
    # todo implement a "smarter" ensemble model, for now just implement one that averages

    # Ensemble model
    hyper_tuned_model, model = fitter(train_site, train_gt, val_site, val_gt)
    train_pred = model(train_site)
    val_pred = hyper_tuned_model(val_site)
    test_pred = model(test)

    return train_pred, val_pred, test_pred


def ensemble_models(preds: [pd.DataFrame], ensemble_type: Ensemble_Type = Ensemble_Type.AVERAGE):
    assert all([preds[i].size == preds[i + 1].size for i in range(0, len(preds) - 1)]), \
        'Sizes of local and global prediction dfs dont match!'
    assert all([(preds[i].site_id == preds[i + 1].site_id).all() for i in range(0, len(preds) - 1)]), \
        'Mismatch between local and global site columns!'

    if ensemble_type == Ensemble_Type.AVERAGE:
        final_pred = pd.DataFrame()
        for idx, pred in enumerate(preds):
            site_id_col = pred.site_id
            date_col = pred.issue_date
            if final_pred.empty:
                final_pred = pred
            else:
                final_pred = final_pred.drop(columns=['site_id', 'issue_date'])

                pred = pred.drop(columns=['site_id', 'issue_date'])

                final_pred = pred.add(final_pred, fill_value=0)
                if idx == (len(preds) - 1):  # last iteration
                    final_pred = final_pred / len(pred)  # nromalize with the length to get the average
                final_pred['site_id'] = site_id_col
                final_pred['issue_date'] = date_col

                # Reorder columns
                cols = final_pred.columns.to_list()
                cols = cols[-1:] + cols[:-1]
                cols = cols[-1:] + cols[:-1]
                final_pred = final_pred[cols]

    # todo implement other ensemble types of models
    return final_pred
