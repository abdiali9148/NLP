import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def evaluate(y_true, y_predict):
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    mae = mean_absolute_error(y_true, y_predict)
    rho, p_val = spearmanr(y_true, y_predict)
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Spearman_rho": float(rho)
    }