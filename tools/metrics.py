def MCRMSE(y_pred, y_true):
    assert len(y_pred.shape) == 3
    return ((y_true - y_pred)**2)\
        .mean(axis=2)\
        .squeeze()\
        .sqrt()\
        .mean(axis=1)\
        .mean()
