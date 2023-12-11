import pandas as pd
from models.models import general_pcr_fitter, xgboost_fitter, k_nearest_neighbors_fitter
from enum import Enum
import numpy as np

class Ensemble_Type(Enum):
    AVERAGE = 0
    BEST_PREDICTION = 1

def gen_basin_preds(train_site: pd.DataFrame, train_gt: pd.DataFrame, val_site: pd.DataFrame, val_gt: pd.DataFrame,
                    test: pd.DataFrame, fitter=xgboost_fitter) -> tuple:
    #if len(model_fitters) != 1:
    #    raise NotImplementedError('Error - not implemented yet for ensembles!')

    # todo reimplement ensembling so it happens elsewhere, so you can still store the individual model predictions
    # todo implement a "smarter" ensemble model, for now just implement one that averages

    # Ensemble model
    hyper_tuned_model, model = fitter(train_site, train_gt, val_site, val_gt)
    train_pred = model(train_site)
    val_pred = hyper_tuned_model(val_site)
    test_pred = model(test)

    return train_pred, val_pred, test_pred

def ensemble_models(preds: dict, ensemble_name: str, ensemble_type:  Ensemble_Type = Ensemble_Type.BEST_PREDICTION):
    # todo make these asserts work for dictionary
    pred_keys = list(preds.keys())

    assert all([preds[pred_keys[i]].size == preds[pred_keys[i + 1]].size for i in range(0, len(pred_keys) - 1)]), \
        'Sizes of local and global prediction dfs dont match!'

    assert all([(preds[pred_keys[i]].site_id == preds[pred_keys[i + 1]].site_id).all() for i in range(0, len(pred_keys) - 1)]), \
        'Mismatch between local and global site columns!'

    ordered_site_ids = preds[pred_keys[0]].site_id.drop_duplicates().tolist()

    if ensemble_type == Ensemble_Type.AVERAGE:
        final_pred = pd.DataFrame()
        for model_name, pred in preds.items():
            site_id_col = pred.site_id
            date_col = pred.issue_date
            if final_pred.empty:
                final_pred = pred
            else:
                final_pred = final_pred.drop(columns=['site_id', 'issue_date'])

                pred = pred.drop(columns=['site_id', 'issue_date']) / len(pred)

                final_pred = pred.add(final_pred, fill_value=0)
                final_pred['site_id'] = site_id_col
                final_pred['issue_date'] = date_col



                # Reorder columns
                cols = final_pred.columns.to_list()
                cols = cols[-1:] + cols[:-1]
                cols = cols[-1:] + cols[:-1]
                final_pred = final_pred[cols]
    elif ensemble_type == Ensemble_Type.BEST_PREDICTION:
        keys = list(preds.keys())
        final_pred = pd.DataFrame()
        for site_id in preds[keys[0]].site_id.unique():
            site_id_preds = {model: pred[pred.site_id==site_id] for model, pred in preds.items()}
            site_keys = list(site_id_preds.keys())

            # Find best prediction by looking through the validation loss
            losses = [(pd.read_csv(f'{model}_{site_id}_avg_q_losses.csv')).val[0] for model, _ in preds.items()]
            best_site_idx = np.argmin(losses)
            best_pred = site_id_preds[site_keys[best_site_idx]]

            final_pred = pd.concat((final_pred, best_pred))
            print(site_keys[best_site_idx])

    final_pred = final_pred.reset_index(drop=True)

    final_pred = final_pred.groupby(final_pred.issue_date.dt.year) \
        .apply(lambda x: x.sort_values(['site_id', 'issue_date']))    # todo implement other ensemble types of models
    return {f'{ensemble_name}': final_pred}
