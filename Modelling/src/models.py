from sklearn import ensemble
from sklearn import  linear_model
from sklearn import  svm

from . import global_var


MODELS = {
    "lin_reg": linear_model.LinearRegression(n_jobs=-1,),

    "ridge": linear_model.Ridge(
        max_iter=10000,
        random_state=global_var.SEED,
    ),
    
    "lasso": linear_model.Lasso(
        max_iter=10000,
        random_state=global_var.SEED,
    ),

    "svr": svm.SVR(),

    "en_rf": ensemble.RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        min_samples_split=14,
        max_leaf_nodes=14,
        n_jobs=-1,
        random_state=global_var.SEED,
    ),

    "en_et": ensemble.ExtraTreesRegressor(
        n_estimators=150,
        max_depth=5,
        min_samples_split=14,
        max_leaf_nodes=14,
        n_jobs=-1,
        random_state=global_var.SEED,
    ),

    "en_adaboost": ensemble.AdaBoostRegressor(
        n_estimators=150,
        random_state=global_var.SEED,
    ),

    "en_gradboost": ensemble.GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=8,
        max_leaf_nodes=14,
        random_state=global_var.SEED,
    ),
}
