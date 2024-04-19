"""Hierarchical Time Series Forecast Reconciliation."""
__description__ = "Time Series Forecast Combination"
__author__ = "Amine Raboun - amineraboun@github.io"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from typing import Dict, Optional

from .forecast import Forecast
from .model_select import ForecastModelSelect

# Plotting packages & configuration
from forecast_combine.utils.plotting import plot_series
import matplotlib.pyplot as plt

class ForecastReconciler:
    """
    ForecastReconciler class to apply the hiearachical reconciliation of the child node forecast with the total forecast.

    Parameters:
    -----------
    - forecasters_d (Dict[str, any]): Dictionary of forecasters to reconcile.
    - S (pd.DataFrame): Summation matrix.
    - method (str): Reconciliation method to use. Default is 'mint'. Available methods are 
        - ols: Ordinary Least Squares
        - wls: Weighted Least Squares
        - mint: Minimum Trace
        - td: Top-Down        
        - bu: Bottom-Up

    Attributes:
    -----------
    - _forecasters_d (Dict[str, any]): Dictionary of forecasters to reconcile.
    - _Total (str): Name of the total forecast.
    - _forecasters_order (List[str]): Order of the forecasters.
    - _S (np.array): Summation matrix.
    - _method (str): Reconciliation method to use.
    - _historical_values_d (Dict[str, pd.Series]): Dictionary of historical values for each forecaster.
    - _historical_errors_d (Dict[str, pd.DataFrame]): Dictionary of historical errors for each forecaster.
    - _weights (Dict[str, np.array]): Dictionary of weights for each forecaster.

    Methods:
    --------
    - fit(refit_models: bool = False, reconciliation_method: Optional[str] = None) -> None:
        Prepare the reconciliation weights or parameters based on historical data.
        Optionally refit models.
    - predict(X: Dict[str, pd.DataFrame] = None, coverage: float = 0.9, reconciliation_method: Optional[str] = None, verbose: bool = False) -> pd.DataFrame:
        Apply the reconciliation method to the forecasts.
    - update(new_y:Dict[str, pd.Series], new_X: Dict[str, pd.DataFrame] = None, reconciliation_method: Optional[str] = None, coverage: float = 0.9, refit: bool = False, reevaluate:bool = False, verbose: bool = False) -> pd.DataFrame:
        Update the forecast with new data.
    - reconcile_preds(forecast_data: Dict[str, pd.DataFrame], forecast_intervals: Dict[str, pd.DataFrame], reconciliation_method: Optional[str] = None) -> pd.DataFrame:
        Apply the reconciliation method to the forecasts.
    - plot_predict(reconciled_preds, reconciled_intervals, interval_label: str = 'CI', title: str = 'Reconciled Predictions') -> None:
        Plot the reconciled predictions and intervals.
    
    Raises:
    -------
    - AssertionError: If the number of forecasters does not match the number of rows in S.
    - AssertionError: If the number of columns in S does not match the number of the child forecasters.
    - AssertionError: If the index values of S do not match the keys in forecasters_d.
    - AssertionError: If the reconciliation method is invalid.
        
    """
    def __init__(self, 
                 forecasters_d: Dict[str, any], 
                 S: pd.DataFrame, 
                 method: str = 'mint',
                 ):
        """ Inintialize the ForecastReconciler class."""
        assert S.shape[0] == len(forecasters_d), "Number of forecasters must match number of rows in S."
        assert S.shape[1] == S.shape[0]-1, "Number of columns in S must match number of the child forecasters."
        assert all([s in forecasters_d.keys() for s in S.index]), "Index values of S must match keys in forecasters_d."
        self._methods_l = ['ols', 'wls', 'td', 'mint', 'bu']
        assert method in self._methods_l, "Invalid reconciliation method."
        self._forecasters_d = forecasters_d
        self._Total = S.index[0]
        self._forecasters_order = S.index
        self._S = S.values
        self._method = method
        self._historical_values_d = None
        self._historical_errors_d = None
        self._weights = None
   
    def fit(self,
            refit_models: bool = False,
            reconciliation_method: Optional[str] = None
            ) -> None:
        """
        Prepare the reconciliation weights or parameters based on historical data.
        Optionally refit models.
        """
        if refit_models:
            for forecaster in self._forecasters_d.values():
                forecaster.fit()
        if reconciliation_method is None:
            reconciliation_method = self._method

        if reconciliation_method in ['wls', 'mint']:
            print(f"{reconciliation_method} reconciliation method requires the forecasters prediction errors variance-covariance matrix ...")
            self._historical_errors_d = self.__compute_hist_errors()
            historical_errors = pd.concat(self._historical_errors_d).reset_index(0).rename(columns={'level_0':'nodes'})

            if reconciliation_method == 'wls':
                _wght = {}
                for horizon, group in historical_errors.groupby('horizon'):
                    _var = group.pivot(index='cutoff', columns='nodes', values='error').var()
                    _wght[horizon] = np.diag(1 / _var.loc[self._forecasters_order])
                self._weights = _wght
            elif reconciliation_method == 'mint':
                _wght = {}
                for horizon, group in historical_errors.groupby('horizon'):
                    ordered_group = group.pivot(index='cutoff', columns='nodes', values='error')[self._forecasters_order]
                    _wght[horizon] = pinv(ordered_group.cov().values)
                self._weights = _wght
        elif reconciliation_method == 'td':
            self._historical_values_d = self.__compute_hist_values()
            historical_values_df = pd.concat(self._historical_values_d, axis=1)[self._forecasters_order]
            historical_values_prop = (historical_values_df.iloc[:, 1:]*self._S[0]).div(historical_values_df[self._Total], axis=0)
            historical_values_prop_mean = historical_values_prop.mean().to_dict()
            historical_values_prop_mean[self._Total] = 1
            self._weights = historical_values_prop_mean
        else:
            pass  # No weights needed for 'bu' or 'ols'
   
    def predict(self, 
                X: Dict[str, pd.DataFrame] = None,
                coverage: float = 0.9,
                reconciliation_method: Optional[str] = None,
                verbose: bool = False
                ) -> pd.DataFrame:
        """
        Apply the reconciliation method to the forecasts.
        
        Parameters:
        -----------
            X (Optional[pd.DataFrame]): 
                DataFrame containing independent variables for prediction, if needed.
            coverage (float):
                Confidence interval for the forecast quantiles.
            reconciliation_method (Optional[str]):
                Override the default method for this prediction.
        
        Returns:
        --------
            reconciled_preds (Dict[str, pd.Series]):
                Dictionary of reconciled forecasts.
            reconciled_intervals (Dict[str, pd.DataFrame]):
                Dictionary of reconciled forecast intervals.
        """
        assert reconciliation_method in self._methods_l, "Invalid reconciliation method."
        if verbose:
            print("Computing the Hiearchical models forecasts ...")
        forecast_data = {}
        forecast_intervals ={}
        for name, forecaster in self._forecasters_d.items():
            if X is None:
                forecster_X = None
            else:
                forecster_X = X[name] if name in X.keys() else None
            if verbose:
                print(f"\tComputing forecasts for {name} ...")
            forecast_data[name], forecast_intervals[name] = forecaster.predict(X=forecster_X, coverage=coverage)

        if verbose:
            print("\nReconciling forecasts ...")
        return self.reconcile_preds(forecast_data, forecast_intervals, reconciliation_method)
    
    def update(self,
               new_y:Dict[str, pd.Series],
               new_X: Dict[str, pd.DataFrame] = None,
               reconciliation_method: Optional[str] = None,
               coverage: float = 0.9, 
               refit: bool = False, 
               reevaluate:bool = False, 
               verbose: bool = False
              ) -> pd.DataFrame: 
        """
        Update the forecast with new data.

        Parameters:
        -----------
            newdata (pd.DataFrame): 
                New data to update the forecast.
            reconciliation_method (Optional[str]):
                Override the default method for this prediction.
            refit (bool):
                If True, refit the models before updating the forecast.

        Returns:
        --------
        Returns:
        --------
            reconciled_preds (Dict[str, pd.Series]):
                Dictionary of reconciled forecasts.
            reconciled_intervals (Dict[str, pd.DataFrame]):
                Dictionary of reconciled forecast intervals.
        """        
        assert reconciliation_method in self._methods_l, "Invalid reconciliation method."   
        if verbose:
            print("Computing the Hiearchical models forecasts ...")
        forecast_data = {}
        forecast_intervals = {}
        for name, forecaster in self._forecasters_d.items():
            forecaster_y = new_y[name] 
            if new_X is None:
                forecster_X = None
            else:
                forecster_X = new_X[name] if name in new_X.keys() else None
            if verbose:
                print(f"\tComputing forecasts for {name} ...")
            forecast_data[name], forecast_intervals[name] = forecaster.update(new_y=forecaster_y, 
                                                                              new_X=forecster_X,
                                                                              coverage=coverage,
                                                                              refit = refit,
                                                                              reevaluate = reevaluate)
            
        if verbose:
            print("\nReconciling forecasts ...")
        return self.reconcile_preds(forecast_data, forecast_intervals, reconciliation_method)

    def reconcile_preds(self, 
                forecast_data: Dict[str, pd.DataFrame],
                forecast_intervals: Dict[str, pd.DataFrame],
                reconciliation_method: Optional[str] = None
                ) -> pd.DataFrame:
        """
        Apply the reconciliation method to the forecasts.
        
        Parameters:
        -----------
            forecast_data (Dict[str, pd.DataFrame]): 
                Dictionary of forecasts for each series.
        
        Returns:
        --------
            reconciled_preds (Dict[str, pd.Series]):
                Dictionary of reconciled forecasts.
            reconciled_intervals (Dict[str, pd.DataFrame]):
                Dictionary of reconciled forecast intervals.           
            
        """
        if reconciliation_method is None:
            reconciliation_method = self._method
        
        if (reconciliation_method in ['wls', 'mint', 'td']):
            self.fit(reconciliation_method=reconciliation_method)
        
        y_preds = np.vstack([forecast_data[name] for name in self._forecasters_order])
        y_preds_lower = np.vstack([forecast_intervals[name].xs('lower', level =2, axis=1).squeeze() for name in self._forecasters_order])
        y_preds_upper = np.vstack([forecast_intervals[name].xs('upper', level =2, axis=1).squeeze() for name in self._forecasters_order])
        
        # Reconcile forecasts and intervals using the same weights and matrices
        reconciled_forecasts = self.__apply_reconciliation(y_preds, reconciliation_method)
        reconciled_lower_bounds = self.__apply_reconciliation(y_preds_lower, reconciliation_method)
        reconciled_upper_bounds = self.__apply_reconciliation(y_preds_upper, reconciliation_method)

        # Return the reconciled forecasts and intervals
        reconciled_preds ={}
        reconciled_intervals = {}
        dates = forecast_data[self._Total].index
        for i, name in enumerate(self._forecasters_order):
            reconciled_preds[name] = pd.Series(reconciled_forecasts[i], index = dates).rename(name)
            _interval = pd.DataFrame(np.vstack([reconciled_lower_bounds[i], reconciled_upper_bounds[i]]).T, 
                                                      index = dates, columns = ['lower', 'upper'])
            _interval.columns = forecast_intervals[name].columns
            reconciled_intervals[name] = _interval        
        return reconciled_preds, reconciled_intervals
    
    def plot_predict(self, 
                     reconciled_preds,
                     reconciled_intervals,
                     interval_label: str = 'CI', 
                    title: str = 'Reconciled Predictions',
                    ):
        """
        Plot the reconciled predictions and intervals.

        Parameters:
        -----------
            reconciled_preds (Dict[str, pd.Series]): 
                Dictionary of reconciled forecasts.
            reconciled_intervals (Dict[str, pd.DataFrame]):
                Dictionary of reconciled forecast intervals.
            interval_label (str):
                Label for the confidence interval.
            title (str):
                Title of the plot.

        Returns:
        --------
            fig (plt.Figure):
                Plot figure.
            axes (plt.Axes):
                Plot axes.
        """
        self._historical_values_d = self.__compute_hist_values()
        nplots = len(reconciled_preds)
        ncols = 2
        nrows = nplots // ncols + nplots % ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6*nrows))
        axes = axes.flatten()
        for i, name in enumerate(reconciled_preds.keys()):
            y_i = self._historical_values_d[name]
            ypred_i = reconciled_preds[name]
            y_i = y_i.loc[y_i.index<ypred_i.index[0]]
            zoom_y_train = y_i.iloc[-3*len(ypred_i):]
            plot_series(zoom_y_train,
                        ypred_i,
                        labels =['y', 'y_pred'],
                        pred_interval= reconciled_intervals[name],
                        interval_label =interval_label,
                        title =name,
                        ax=axes[i], 
                        )
        fig.tight_layout()
        fig.suptitle(title, size="xx-large")
        fig.subplots_adjust(top=0.9)        
        return fig, axes
    
    def __compute_hist_errors(self, verbose=False):
        if self._historical_errors_d is None:                
            if verbose:
                print("\nComputing the prediction errors for all forecasters ...")
            _d = {}
            for k, forecaster in self._forecasters_d.items():
                if verbose:
                    print(f"\tComputing prediction errors for {k} ...")
                _d[k] = forecaster.get_pred_errors()
            self._historical_errors_d = _d
        return self._historical_errors_d
    def __compute_hist_values(self, verbose=False):    
        if self._historical_values_d is None:
            if verbose:
                print("Computing the historical values for all forecasters ...")
            _d = {}
            for k, forecaster in self._forecasters_d.items():
                if isinstance(forecaster, Forecast):
                    _d[k] = forecaster._y
                elif isinstance(forecaster, ForecastModelSelect):
                    _d[k] = list(forecaster.LF_d.values())[0]._y
            self._historical_values_d = _d
        return self._historical_values_d
    
    def __apply_reconciliation(self, y_preds: np.array, reconciliation_method: str) -> pd.DataFrame:
        """
        Apply the reconciliation method to the forecasts.
        """
        if reconciliation_method == 'td':
            total_forecast = y_preds[0]
            reconciled_forecasts_d = {name: total_forecast.T * self._weights[name] for name in self._forecasters_order}
            reconciled_forecasts = reconciled_forecasts_d.values()
            reconciled_forecasts = np.array([total_forecast.T * self._weights[name] for name in self._forecasters_order])  
        elif reconciliation_method == 'bu':
            reconciled_forecasts = self._S @ y_preds[1:, :]
        
        elif reconciliation_method == 'ols':
            M = self._S @ pinv(self._S.T @ self._S) @ self._S.T
            reconciled_forecasts = M  @ y_preds
        
        elif reconciliation_method in ['wls', 'mint']:
            reconciled_forecasts = pd.DataFrame()
            for horizon in self._weights.keys():
                M = self._S @ pinv(self._S.T @ self._weights[horizon] @ self._S) @ self._S.T @ self._weights[horizon]
                reconciled_forecasts = M @ y_preds

        return reconciled_forecasts