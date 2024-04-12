
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import pinv

class ForecastReconciler:
    def __init__(self, forecasters, S, method ='mint'):
        """
        Initializes the ForecastReconciler class.
        
        Parameters:
        - forecasters (dict): Dictionary of forecaster objects for each series.
        - S (numpy.array): Aggregation matrix indicating how lower levels aggregate into the total.
        - method (str): Method of reconciliation ('ols', 'wls', 'td', 'mint', 'bu').
        """
        self.forecasters = forecasters
        self.S = S
        self.method = method

        self.historical_values_d = {k:lf.y for k,lf in forecasters.items()}
        self.historical_errors_d = {k:lf.get_historical_errors for k,lf in forecasters.items()}
        self.weights = None

    def fit(self, historical_data):
        """
        Prepare the reconciliation weights or parameters based on historical data.
        
        Parameters:
        - historical_data (pd.DataFrame): DataFrame containing historical forecasts and actuals.
        """
        historical_data_df = pd.DataFrame({key: historical_data[key] for key in self.forecasters.keys()})
        
        historical_errors = {key: historical_data[key] - historical_data['Actuals']
                             for key in self.forecasters.keys()}
        historical_errors_df = pd.DataFrame(historical_errors)
        
        if self.method == 'wls':
            error_variances = historical_errors_df.var()
            self.weights = 1 / error_variances
        elif self.method == 'mint':
            cov_matrix = np.cov(historical_errors_df.T, bias=True)
            self.weights = pinv(cov_matrix)
        elif self.method == 'td':
            self.weights = historical_data_df.mean() / historical_data_df['Total'].mean()
        else:
            pass
        return self

    def predict(self, forecast_data):
        """
        Apply the reconciliation method to the forecasts.
        
        Parameters:
        - forecast_data (pd.DataFrame): DataFrame containing forecast data for each series.
        
        Returns:
        - pd.DataFrame: Reconciled forecasts.
        """
        forecasts = np.array([forecast_data[key] for key in self.forecasters.keys()])
        if self.method == 'ols':
            # Simple OLS regression
            X = np.vstack([forecasts, np.ones(forecasts.shape[1])]).T
            y = self.S @ forecasts
            model = sm.OLS(y, X).fit()
            reconciled_forecasts = model.predict(X)
        
        elif self.method == 'wls':
            # Weighted least squares
            X = np.vstack([forecasts, np.ones(forecasts.shape[1])]).T
            y = self.S @ forecasts
            model = sm.WLS(y, X, weights=self.weights).fit()
            reconciled_forecasts = model.predict(X)
        
        elif self.method == 'td':
            # Top-Down using historical proportions
            total_forecast = forecast_data['Total']
            reconciled_forecasts = {key: total_forecast * self.weights[key] for key in self.forecasters.keys()}
        
        elif self.method == 'mint':
            # MINT reconciliation
            M = self.weights @ self.S.T @ pinv(self.S @ self.weights @ self.S.T)
            reconciled_forecasts = M @ forecasts

        elif self.method == 'bu':
            # Bottom-Up method
            # Aggregate forecasts from the lowest level specified in the matrix S
            reconciled_forecasts = np.dot(self.S, forecasts)
        
        return pd.DataFrame(reconciled_forecasts, index=self.forecasters.keys()).T

# Example usage:
forecasters = {
    'Total': None,  # Placeholder for actual forecaster objects
    'Lower1': None,
    'Lower2': None
}
S = np.array([[1, 1], [1, 0], [0, 1]])  # Example aggregation matrix
reconciler = ForecastReconciler(forecasters, S, method='bu')
historical_data = pd.DataFrame(...)  # Placeholder for actual historical data
forecast_data = pd.DataFrame(...)  # Placeholder for forecast data
reconciler.fit(historical_data)
reconciled_forecasts = reconciler.predict(forecast_data)
