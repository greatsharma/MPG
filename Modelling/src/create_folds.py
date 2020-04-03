import os
import argparse
import pandas as pd
from sklearn import model_selection

from . import utils
from . import global_var
from .cross_validation import CrossValidation


ap = argparse.ArgumentParser()
ap.add_argument('-s', '--shuffle', required=True,
                help='1 for shuffle otherwise 0')
args = vars(ap.parse_args())

shuffle = utils.arg2bool(args['shuffle'])

INPUT_PATH = os.environ.get('INPUT_PATH')
df = pd.read_csv(INPUT_PATH + 'train.csv')

print('creating cv folds...\n')

cv = CrossValidation(shuffle=shuffle, num_folds=global_var.NUM_FOLDS,
                     problem_type="single_column_regression", random_state=global_var.SEED)
df = cv.split(df, target_cols=["target"])

df.to_csv(INPUT_PATH + "train_folds.csv", index=False)