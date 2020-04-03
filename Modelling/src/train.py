import os
import random
import joblib
import pandas as pd

from . import global_var
from . import models
from . import metrics
from . import losses
from . import logger


INPUT_PATH = os.environ.get('INPUT_PATH')
META_PATH = os.environ.get('META_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')
MODEL = os.environ.get('MODEL')
PIPE = os.environ.get('PIPE')

model_class = models.MODELS[MODEL].__class__.__name__

df = pd.read_csv(INPUT_PATH + 'train_folds.csv')

all_data_pipe = joblib.load(META_PATH + 'all_data_pipe.pkl')
pipe = joblib.load(META_PATH + f'{PIPE}.pkl')

logger.log(logger.file_obj, msg=f'\t\tTraining {model_class} on {PIPE} ...\n', date_time=True,)
print(f'\nTraining {model_class} on {PIPE} ...\n')

cv_metric = []
cv_loss = []
leaderboard = pd.read_csv(INPUT_PATH + 'leaderboard.csv')
model_predictions = pd.DataFrame(columns=['fold', 'true', 'pred'])

for fold in range(global_var.NUM_FOLDS):
    logger.log(logger.file_obj, msg=f'\n----------Fold {fold+1}--------------')
    print(f'----------Fold {fold+1}--------------')

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["target", "kfold"], axis=1)
    valid_df = valid_df.drop(["target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    train_df = all_data_pipe.transform(train_df)
    train_df = pipe.fit_transform(train_df, ytrain)
    if type(train_df) != pd.DataFrame:
        train_df = pd.DataFrame(train_df)

    joblib.dump(pipe, META_PATH + f'{PIPE}_cv{fold+1}.pkl')

    estimator = models.MODELS[MODEL]
    estimator.fit(train_df.values, ytrain)

    valid_df = all_data_pipe.transform(valid_df)
    valid_df = pipe.transform(valid_df)
    if type(valid_df) != pd.DataFrame:
        valid_df = pd.DataFrame(valid_df)

    ypreds = estimator.predict(valid_df.values)
    model_predictions = model_predictions.append(pd.DataFrame({
        'fold':fold+1, 'true':yvalid, 'pred':ypreds}), ignore_index=True)

    metric = metrics.RegressionMetrics()(global_var.METRIC, yvalid, ypreds)
    cv_metric.append(metric)
    logger.log(logger.file_obj, msg=f'\n{global_var.METRIC}: {metric}')
    print(f'{global_var.METRIC}: {metric}')

    loss = losses.RegressionLoss()(global_var.LOSS, yvalid, ypreds)
    cv_loss.append(loss)
    logger.log(logger.file_obj, msg=f'\n{global_var.LOSS}: {loss}')
    print(f'{global_var.LOSS}: {loss}')

    MODEL_NAME = f"{MODEL}_{PIPE}_cv{fold+1}_.pkl"
    joblib.dump(estimator, MODEL_PATH + MODEL_NAME)
    
    leaderboard = leaderboard.append({
        'id':random.randint(a=1000, b=100000000), 'estimator': model_class, 'pipe': PIPE,
        'total_folds': global_var.NUM_FOLDS, 'fold': fold+1, 'loss': loss, 'metric': metric,
    }, ignore_index=True)

for f in os.listdir(INPUT_PATH):
    if f == f'df_{PIPE}.csv':
        break
else:
    train_df.to_csv(INPUT_PATH + f'df_{PIPE}.csv', index=False)

avg_metric = sum(cv_metric) / global_var.NUM_FOLDS
avg_loss =  sum(cv_loss) / global_var.NUM_FOLDS

logger.log(logger.file_obj, msg=f'\n\ncv avg {global_var.METRIC}: {avg_metric}')
print(f'\ncv avg {global_var.METRIC}: {avg_metric}')

logger.log(logger.file_obj, msg=f'\ncv avg {global_var.LOSS}: {avg_loss}')
print(f'cv avg {global_var.LOSS}: {avg_loss}')

joblib.dump(cv_metric, MODEL_PATH + f'metric_{MODEL}_{PIPE}_cv{global_var.NUM_FOLDS}_.pkl')
joblib.dump(cv_loss, MODEL_PATH + f'loss_{MODEL}_{PIPE}_cv{global_var.NUM_FOLDS}_.pkl')

leaderboard.to_csv(INPUT_PATH + "leaderboard.csv", index=False)
model_predictions.to_csv(INPUT_PATH + f"pred_{model_class}.csv", index=False)

logger.file_obj.close()