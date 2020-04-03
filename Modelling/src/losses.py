from sklearn import metrics as skmetric


class Loss:

    def __init__(self, losses):
        self.losses = losses

    def __call__(self, loss, y_true, y_pred):
        raise NotImplementedError


class ClassificationLoss(Loss):

    def __init__(self):
        super(ClassificationLoss, self).__init__({
            "logloss": self._logloss
        }
    )
        
    def __call__(self, loss, y_true, y_pred, y_proba):
        if loss not in self.losses:
            raise Exception("loss not implemented")
        elif loss == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for logloss")
        
    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetric.log_loss(y_true=y_true, y_pred=y_pred)

        
class RegressionLoss(Loss):

    def __init__(self):
        super(RegressionLoss, self).__init__({
            "mse": self._mse,
            "mae": self._mae,
            "msle": self._msle,
            "median_absolute_error": self._mdae
        }
    )
    
    def __call__(self, loss, y_true, y_pred):
        if loss not in self.losses:
            raise Exception("loss not implemented")
        else:
            return self.losses[loss](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetric.mean_squared_error(y_true, y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetric.mean_absolute_error(y_true, y_pred)
        
    @staticmethod
    def _msle(y_true, y_pred):
        return skmetric.mean_squared_log_error(y_true, y_pred)

    @staticmethod
    def _mdae(y_true, y_pred):
        return skmetric.median_absolute_error(y_true, y_pred)
