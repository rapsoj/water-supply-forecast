import pandas as pd
from models.models import general_pcr_fitter
from models.models import xgboost_fitter


def gen_basin_preds(train_site: pd.DataFrame, train_gt: pd.DataFrame, val_site: pd.DataFrame, val_gt: pd.DataFrame,
                    test: pd.DataFrame, model_fitters=(xgboost_fitter,)) -> tuple:
    if len(model_fitters) != 1:
        raise NotImplementedError('Error - not implemented yet for ensembles!')

    fitter = model_fitters[0]

    hyper_tuned_model, model = fitter(train_site, train_gt, val_site, val_gt)
    train_pred = model(train_site)
    val_pred = hyper_tuned_model(val_site)
    test_pred = model(test)

    return train_pred, val_pred, test_pred
