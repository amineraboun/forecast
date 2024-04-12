import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error
)

def evaluate_metrics(y_true, y_pred):
    """
    Computes performance metrics for the fitted curve.
    The performance metrics include 
        RMSE: Root Mean Squared Error
        MAE: Mean Absolute Error
        MAPE: Mean Absolute Percentage Error
        R2: R-squared
        MedianAE: Median Absolute Error
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)),
        'R2': r2_score(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred)
    }
    return pd.Series(metrics)

def summary_perf(insample_result_df, 
			 grouper, 
			 y_true_col = 'y_true',
			 y_pred_col = 'y_pred',
			):
    """
	Compute summary performance metrics for a given forecast.
        RMSE: Root Mean Squared Error
        MAE: Mean Absolute Error
        MAPE: Mean Absolute Percentage Error
        R2: R-squared
        MedianAE: Median Absolute Error

	Parameters:
    -----------
	    insample_result_df (DataFrame): 
            A DataFrame containing the forecast results.
		grouper (str): 
            The column name to group the forecast results for computing summary metrics.
		y_true_col (str, optional):
            The column name representing the true values. Default is 'y_true'.
		y_pred_col (str, optional): 
            The column name representing the predicted values. Default is 'y_pred'.

	Returns:
    --------
		DataFrame: A DataFrame containing summary performance metrics.
	"""

    horizon_metrics = insample_result_df.groupby(grouper).apply(lambda x: evaluate_metrics(x[y_true_col], x[y_pred_col]))
    return horizon_metrics