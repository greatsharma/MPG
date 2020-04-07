import os
import joblib
import pandas as pd

from .... import global_var
from .... import metrics
from .... import losses


INPUT_PATH = os.environ.get('INPUT_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')
ESTIMATOR = os.environ.get('ESTIMATOR')

df = pd.read_csv(INPUT_PATH + 'test.csv')
ytrue = df.target.values
df.drop('target', axis=1, inplace=True)

predictions = None
for fold in range(global_var.NUM_FOLDS):
    pipe = joblib.load(MODEL_PATH + f'{ESTIMATOR}_cv{fold+1}_.pkl')
    ypreds = pipe.predict(df)

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
submission.to_csv(INPUT_PATH + f"submission_{ESTIMATOR}.csv", index=False)