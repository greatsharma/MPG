import pandas as pd
from sklearn import model_selection


SUPPORTED_PROBLEM_TYPE = [
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "single_column_regression",
    "multi_column_regression",
    "holdout_"
]


class CrossValidation:
    def __init__(
            self,
            shuffle,
            num_folds=5,
            problem_type="binary_classification",
            multilabel_delimiter=",",
            random_state=42,
            verbose=True
        ):
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state
        self.verbose = verbose
    
    def split(self, X, target_cols, copy=True):
        if self.problem_type not in SUPPORTED_PROBLEM_TYPE:
            raise ValueError("Problem type not supported")

        num_targets = len(target_cols)

        df = (
            X.copy(deep=True)
            if copy else
            X
        )

        if self.shuffle is True:
            df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        df["kfold"] = -1

        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = target_cols[0]
            unique_values = df[target].nunique()
            if unique_values == 1:
                raise Exception(f"Only one unique value found for {self.problem_type}")
            else:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target].values)):
                    if self.verbose:
                            print(f'\tfold_{fold+1}\t', len(train_idx), len(val_idx))
                    df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type == "multilabel_classification":
            if num_targets != 1:
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            targets = df[target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=targets)):
                if self.verbose:
                        print(f'\tfold_{fold+1}\t', len(train_idx), len(val_idx))
                df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_column_regression", "multi_column_regression"):
            if num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            if num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception(f"Invalid number of targets for {self.problem_type}")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
                if self.verbose:
                        print(f'\tfold_{fold+1}\t', len(train_idx), len(val_idx))
                df.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(df) * holdout_percentage / 100)
            df.loc[:len(df) - num_holdout_samples, "kfold"] = 0
            df.loc[len(df) - num_holdout_samples:, "kfold"] = 1

        return df


if __name__ == "__main__":
    df = pd.DataFrame({'name': ['a','c','b','a','c','a'],
                       'rank':[5,6,5,7,8,6],
                       'id': [13,23,45,'rare',90,67]})
    cv = CrossValidation(shuffle=False, num_folds=3, problem_type="single_column_regression")
    df_split = cv.split(df, target_cols=["rank"])
    print(df.head())
    print(df_split.head())
