# run INPUT_PATH=input/ META_PATH=meta/ MODEL_PATH=models/ python -m src.automl.tpot_checkpoints.20200407_135014_pipe5_gen30_pop30.exported_tpot
import os
import joblib
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.export_utils import set_param_recursive

from .... import global_var
from .... import metrics
from .... import losses

INPUT_PATH = os.environ.get('INPUT_PATH')
META_PATH = os.environ.get('META_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')


all_data_pipe = joblib.load(META_PATH + 'all_data_pipe.pkl')
pipe5 = joblib.load(META_PATH + 'pipe5.pkl')

# Average CV score on the training set was: 0.8889194331157595
exported_pipeline = make_pipeline(
    all_data_pipe,
    pipe5,
    SelectPercentile(score_func=f_regression, percentile=54),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ElasticNetCV(l1_ratio=0.9, tol=0.0001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
df = pd.read_csv(INPUT_PATH + 'train_folds.csv')

cv_metric = []
cv_loss = []
# leaderboard = pd.read_csv(INPUT_PATH + 'leaderboard.csv')

for fold in range(global_var.NUM_FOLDS):
    print(f'----------Fold {fold+1}--------------')

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["target", "kfold"], axis=1)
    valid_df = valid_df.drop(["target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    exported_pipeline.fit(train_df, ytrain)
    ypreds = exported_pipeline.predict(valid_df)

    MODEL_NAME = f"tpot_pipe5_gen30_pop30_cv{fold+1}_.pkl"
    joblib.dump(exported_pipeline, MODEL_PATH + MODEL_NAME)

    metric = metrics.RegressionMetrics()(global_var.METRIC, yvalid, ypreds)
    cv_metric.append(metric)
    print(f'{global_var.METRIC}: {metric}')

    loss = losses.RegressionLoss()(global_var.LOSS, yvalid, ypreds)
    cv_loss.append(loss)
    print(f'{global_var.LOSS}: {loss}')

    # leaderboard = leaderboard.append({
    #     'id':random.randint(a=1000, b=100000000), 'estimator': 'tpot_pipe5_gen30_pop_30', 'pipe': 'pipe5',
    #     'total_folds': global_var.NUM_FOLDS, 'fold': fold+1, 'loss': loss, 'metric': metric,
    # }, ignore_index=True)


avg_metric = sum(cv_metric) / 5
avg_loss =  sum(cv_loss) / 5

print(f'\ncv avg r2: {avg_metric}')
print(f'cv avg MAE: {avg_loss}')

# leaderboard.to_csv(INPUT_PATH + "leaderboard.csv", index=False)
joblib.dump(exported_pipeline, META_PATH + "pipe_tpot_pipe5_gen30_pop30.pkl")