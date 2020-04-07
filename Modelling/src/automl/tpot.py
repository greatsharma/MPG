import os
import sys
import joblib
import pprint
import pandas as pd
from datetime import datetime
from tpot import TPOTRegressor

from .. import global_var
from .. import metrics


INPUT_PATH = os.environ.get('INPUT_PATH')
META_PATH = os.environ.get('META_PATH')
LOGS_PATH = os.environ.get('LOGS_PATH')
GENS = os.environ.get('GENS')
POP_SIZE = os.environ.get('POP_SIZE')
MAX_TIME = os.environ.get('MAX_TIME')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH')
PERIODIC_CHECKPT = os.environ.get('PERIODIC_CHECKPT')
PIPE = os.environ.get('PIPE')

df = pd.read_csv(INPUT_PATH + 'train.csv')
ytrain = df.target
Xtrain = df.drop(['target'], axis=1)

all_data_pipe = joblib.load(META_PATH + 'all_data_pipe.pkl')
Xtrain = all_data_pipe.fit_transform(Xtrain)

if not PIPE is None:
    pipeline = joblib.load(META_PATH + f'{PIPE}.pkl')
    Xtrain = pipeline.fit_transform(Xtrain, ytrain)

del df

if not MAX_TIME is None:
    MAX_TIME = int(MAX_TIME)

if PERIODIC_CHECKPT == "true":
    periodic_checkpoint_folder = CHECKPOINT_PATH
else:
    periodic_checkpoint_folder = None

tpot = TPOTRegressor(generations=int(GENS), population_size=int(GENS), max_time_mins=MAX_TIME, cv=5,
                    scoring='r2', random_state=global_var.SEED, verbosity=3, n_jobs=-1, 
                    periodic_checkpoint_folder=periodic_checkpoint_folder,)

sys.stdout = open(LOGS_PATH, "a")

now = datetime.now()
print('\n\n\n' + '[' + str(now.date()) + "/" + str(now.strftime("%H:%M:%S")) + ']' + f'\tTPOT Begins\n')

print(f"Generations={GENS}\tPopulation_Size={POP_SIZE}\tMax_Time={MAX_TIME}\tPIPE={PIPE}\n")

tpot.fit(Xtrain, ytrain)
pprint.pprint(tpot.fitted_pipeline_)

now = datetime.now()
print('\n' + '[' + str(now.date()) + "/" + str(now.strftime("%H:%M:%S")) + ']' + '\tTPOT Finished')

sys.stdout.close()
sys.stdout = sys.__stdout__

tpot.export(CHECKPOINT_PATH + '/exported_tpot.py')

joblib.dump(tpot.evaluated_individuals_, CHECKPOINT_PATH + '/tpot_evaluated_individuals_.pkl')
joblib.dump(tpot.pareto_front_fitted_pipelines_, CHECKPOINT_PATH + '/tpot_pareto_front_fitted_pipelines_.pkl')
