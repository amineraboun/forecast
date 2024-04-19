import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error
)

def evaluate_metrics(y_true, y_pred):
    """
    Computes performance metrics for the fitted curve. The currently covered performance metrics include 
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - R2: R-squared
    - MedianAE: Median Absolute Error

    Parameters:
    -----------
        y_true (array-like): 
            The true values.
        y_pred (array-like): 
            The predicted values.

    Returns:
    --------
        Series: A series containing the performance metrics.
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
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - R2: R-squared
    - MedianAE: Median Absolute Error


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

def calculate_prediction_interval(historical_errors, y_pred, coverage=0.90):
    """
    Calculate the prediction interval for given predictions based on historical errors.

    Parameters:
    -----------
    historical_errors : array-like
        Array of historical prediction errors (actual - predicted values).
    y_pred : float or array-like
        Predicted value(s) for which the prediction interval is required.
    coverage : float, optional
        Desired coverage of the prediction interval. Default is 0.90.

    Returns:
    --------
    interval : tuple
        A tuple containing the lower and upper bounds of the prediction interval.
    """
    # Calculate the mean and standard deviation of the errors
    sigma_e = historical_errors.groupby('horizon').error.std().sort_index()
    sigma_e.index = y_pred.index

    # Determine the z-score for the desired coverage
    z = norm.ppf((1 + coverage) / 2)

    # Calculate the prediction interval
    lower_bound = y_pred - z * sigma_e
    upper_bound = y_pred + z * sigma_e

    ypred_int = pd.concat([lower_bound, upper_bound], axis=1)
    ypred_int.columns = pd.MultiIndex.from_tuples([(y_pred.name, coverage, 'lower'),
                                                  (y_pred.name, coverage, 'upper')])
    return ypred_int
