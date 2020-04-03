from sklearn import metrics as skmetrics


class Metrics:

    def __init__(self, metric):
        self.metrics = metric

    def __call__(self, metric, y_true, y_pred):
        raise NotImplementedError


class ClassificationMetrics:

    def __init__(self):
        super(ClassificationMetrics, self).__init__({
            "accuracy": self._accuracy,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "auc": self._auc,
        }
    )
        
    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for AUC")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    
        
class RegressionMetrics(Metrics):

    def __init__(self):
        super(RegressionMetrics, self).__init__({
            "r2": self._r2,
            "explained_variance": self._ev_score
        }
    )
    
    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true, y_pred)

    @staticmethod
    def _ev_score(y_true, y_pred):
        return skmetrics.explained_variance_score(y_true, y_pred)