
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def error_metrics(obs, sim):
    """Calculate error metrics."""
    sim = sim.flatten()
    r = np.corrcoef(obs, sim)[0, 1]
    r2 = r ** 2
    mse = mean_squared_error(obs, sim)
    rmse = np.sqrt(mse)
    rrmse = (rmse / np.mean(obs)) * 100
    mae = mean_absolute_error(obs, sim)
    nash = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
    return r, r2, mse, rmse, rrmse, mae, nash
