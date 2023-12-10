import pandas as pd
from models.models import general_pcr_fitter, xgboost_fitter, k_nearest_neighbors_fitter
from enum import Enum
class Ensemble_Type(Enum):
    AVERAGE = 0
    BEST_PREDICTION = 1

def gen_basin_preds(train_site: pd.DataFrame, train_gt: pd.DataFrame, val_site: pd.DataFrame, val_gt: pd.DataFrame,
                    test: pd.DataFrame, model_fitters=(general_pcr_fitter, xgboost_fitter,), ensemble_type: Ensemble_Type=Ensemble_Type.AVERAGE) -> tuple:
    #if len(model_fitters) != 1:
    #    raise NotImplementedError('Error - not implemented yet for ensembles!')

    # todo reimplement ensembling so it happens elsewhere, so you can still store the individual model predictions
    # todo implement a "smarter" ensemble model, for now just implement one that averages
    # Ensemble model
    if ensemble_type == Ensemble_Type.AVERAGE:
        sum_train_pred = pd.DataFrame()
        sum_val_pred = pd.DataFrame()
        sum_test_pred = pd.DataFrame()
        for fitter in model_fitters:

            hyper_tuned_model, model = fitter(train_site, train_gt, val_site, val_gt)
            train_pred = model(train_site)
            val_pred = hyper_tuned_model(val_site)
            test_pred = model(test)
            sum_train_pred = sum_train_pred.add(train_pred, fill_value=0)
            sum_val_pred = sum_val_pred.add(val_pred, fill_value=0)
            sum_test_pred = sum_test_pred.add(test_pred, fill_value=0)
        normalizer = len(model_fitters)
        ave_train_pred = sum_train_pred / normalizer
        ave_val_pred = sum_val_pred / normalizer
        ave_test_pred = sum_test_pred / normalizer

    return ave_train_pred, ave_val_pred, ave_test_pred

def merge_global_local_models(global_pred: pd.DataFrame, local_pred: pd.DataFrame,
                              ensemble_type:  Ensemble_Type = Ensemble_Type.AVERAGE):
    if ensemble_type == Ensemble_Type.AVERAGE:

        assert (global_pred.size == local_pred.size), \
            'Sizes of local and global prediction dfs dont match!'
        assert (global_pred.site_id == local_pred.site_id).all(), \
            'Mismatch between local and global site columns!'

        site_id_col = global_pred.site_id
        date_col = global_pred.issue_date

        global_pred = global_pred.drop(columns=['site_id', 'issue_date'])
        local_pred = local_pred.drop(columns=['site_id', 'issue_date'])

        final_pred = global_pred.add(local_pred, fill_value=0)
        final_pred = final_pred / 2
        final_pred['site_id'] = site_id_col
        final_pred['issue_date'] = date_col

        # Reorder columns
        cols = final_pred.columns.to_list()
        cols = cols[-1:] + cols[:-1]
        cols = cols[-1:] + cols[:-1]
        final_pred = final_pred[cols]
    # todo implement other ensemble types of local+global models
    return final_pred
