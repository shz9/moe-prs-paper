import numpy as np


def compare_scalers(scaler1, scaler2):
    return np.allclose(scaler1.mean_, scaler2.mean_) and \
           np.allclose(scaler1.var_, scaler2.var_) and \
           np.allclose(scaler1.scale_, scaler2.scale_)
