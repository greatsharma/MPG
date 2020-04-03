import os
import joblib
import pandas as pd

from . import global_var
from . import metrics
from . import losses
from . import models


INPUT_PATH = os.environ.get('INPUT_PATH')
META_PATH = os.environ.get('META_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')
MODEL = os.environ.get('MODEL')
PIPE = os.environ.get('PIPE')

print(f'\n\npredicting test data using {models.MODELS[MODEL].__class__.__name__}...')

all_data_pipe = joblib.load(META_PATH + 'all_data_pipe.pkl')

df = pd.read_csv(INPUT_PATH + 'test.csv')
ytrue = df.target.values
df.drop('target', axis=1, inplace=True)

predictions = None
for fold in range(global_var.NUM_FOLDS):
    pipe = joblib.load(META_PATH + f'{PIPE}_cv{fold+1}.pkl')
    
    df_ = all_data_pipe.transform(df)
    df_ = pipe.transform(df_)
    if type(df_) != pd.DataFrame:
        df_ = pd.DataFrame(df_)

    MODEL_NAME = f"{MODEL}_{PIPE}_cv{fold+1}_.pkl"
    estimator = joblib.load(MODEL_PATH + MODEL_NAME)
    
    ypreds = estimator.predict(df_.values)

    if fold == 0:
        predictions = ypreds
    else:
        predictions += ypreds


predictions /= global_var.NUM_FOLDS
metric = metrics.RegressionMetrics()(global_var.METRIC, ytrue, predictions)
loss = losses.RegressionLoss()(global_var.LOSS, ytrue, predictions)

print(f'{global_var.METRIC}: {metric}')
print(f'{global_var.LOSS}: {loss}')

submission = pd.DataFrame(predictions, columns=["pred"])
submission.to_csv(INPUT_PATH + f"submission_{models.MODELS[MODEL].__class__.__name__}.csv", index=False)