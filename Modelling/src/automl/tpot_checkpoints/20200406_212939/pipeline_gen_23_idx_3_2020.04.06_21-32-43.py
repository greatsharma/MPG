import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8778972764720866
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            FeatureAgglomeration(affinity="euclidean", linkage="average"),
            SelectFwe(score_func=f_regression, alpha=0.045)
        ),
        make_pipeline(
            StackingEstimator(estimator=LinearSVR(C=0.1, dual=True, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=1e-05)),
            MinMaxScaler()
        )
    ),
    XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=15, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.6000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
