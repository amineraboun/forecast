""" Timeseries Forecasting with Insample, Validation and Out-of-Sample predictions"""
__description__ = "Time Series Forecast"
__author__ = "Amine Raboun - amineraboun@github.io"

# Filter Warnings
import warnings
warnings.simplefilter('ignore')

##############################################################################
# Import Libraries & default configuration 
##############################################################################
# Standard Inports
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

# Iteration tools
from collections.abc import Iterable
from multiprocessing import Pool
from tqdm import tqdm
import time

# Plotting packages & configuration
from forecast_combine.utils.plotting import plot_series, plot_windows
import matplotlib.pyplot as plt
plt.rc('axes', titlesize='large')    
plt.rc('axes', labelsize='large')   
plt.rc('xtick', labelsize='large')   
plt.rc('ytick', labelsize='large')   
plt.rc('legend', fontsize='large')   
plt.rc('figure', titlesize='x-large') 
import seaborn as sns
sns.set_theme(style='white', font_scale=1)

# Cross Validation tools
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import CutoffSplitter
from sktime.forecasting.model_selection import (
ExpandingWindowSplitter, 
temporal_train_test_split
)

# Performance Metrics
from .utils.metrics import summary_perf, calculate_prediction_interval
from sktime.performance_metrics.forecasting import (
MeanSquaredError,
MeanAbsoluteError, 
MeanAbsolutePercentageError, 
MedianAbsoluteError
)

##############################################################################
# Default values
##############################################################################
# Common Forcasting models
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.statsforecast import (
StatsForecastAutoARIMA,
StatsForecastAutoETS, 
StatsForecastAutoCES,
StatsForecastAutoTheta,
StatsForecastAutoTBATS, 
)
CommonForecastingModels = {
"Naive": NaiveForecaster(),
"Seasonal_Naive": NaiveForecaster(sp = 5),
"AutoARIMA": StatsForecastAutoARIMA(),
"AutoETS": StatsForecastAutoETS(),
"AutoCES": StatsForecastAutoCES(),
"AutoTheta": StatsForecastAutoTheta(),
"AutoTBATS": StatsForecastAutoTBATS(seasonal_periods = 1),
"Prophet": Prophet(),
}

##############################################################################
# Master Class Definition and Plot
##############################################################################
class Forecast(object):
    """
    Forecast class for managing and evaluating forecast models.

    Parameters:
    -----------
    - data (pd.DataFrame): A DataFrame containing the input data for forecasting.
    - depvar_str (str): The column name representing the dependent variable for forecasting.
    - fh (int): The forecast horizon, i.e., the number of periods ahead to forecast.
    - pct_initial_window (float): The percentage of data used as the initial training window.
    - step_length (int): The step size for expanding window cross-validation.
    - forecaster_name (str, optional): The name of the forecasting model. Default is 'Naive'.
    - forecaster (object, optional): The forecasting model object. Default is None, which will use the model corresponding to forecaster_name.
    - exog_l (list, optional): List of exogenous variables for forecasting. Default is None.
    - freq (str, optional): The frequency of the time series data. Default is 'B' (business days).

    Attributes:
    -----------
    - depvar (str): The column name representing the dependent variable for forecasting.
    - exog_l (list): List of exogenous variables for forecasting.
    - freq (str): The frequency of the time series data.
    - forecaster_name (str): The name of the forecasting model.
    - forecaster (object): The forecasting model object.
    - _y (pd.Series): The dependent variable for forecasting.
    - _X (pd.DataFrame): The exogenous variables for forecasting.
    - _fh (ForecastingHorizon): The forecast horizon.
    - _initial_window (int): The initial training window size.
    - _step_length (int): The step size for expanding window cross-validation.
    - _cv (ExpandingWindowSplitter): The cross-validation window.
    - _X_train (pd.DataFrame): The training set of exogenous variables.
    - _X_test (pd.DataFrame): The test set of exogenous variables.
    - _y_train (pd.Series): The training set of the dependent variable.
    - _y_test (pd.Series): The test set of the dependent variable.
    - is_fitted (bool): A flag indicating if the model is fitted.
    - _fitted (ForecastFit): An instance of the ForecastFit class containing the fitted model and insample performance metrics.
    - is_evaluated (bool): A flag indicating if the model is evaluated.
    - _eval (ForecastEval): An instance of the ForecastEval class containing the out-of-sample evaluation results.
    - plot (ForecastPlot): An instance of the ForecastPlot class for plotting utility.

    Methods:
    --------
    - split_procedure_summary(verbose: bool=True) -> dict: Generate a summary of the cross-validation procedure.
    - fit(on: str='all', fh: Optional[ForecastingHorizon]=None) -> ForecastFit: Fit the forecaster and compute insample results.
    - evaluate() -> ForecastEval: Evaluate the forecaster out-of-sample.
    - predict(X: Optional[pd.DataFrame]=None, fh: Optional[ForecastingHorizon]=None, coverage: float=0.9, verbose=False) -> Tuple[pd.DataFrame, pd.DataFrame]: Generate predictions using the fitted model.
    - update(new_y: pd.Series, new_X: Optional[pd.DataFrame]=None, fh: Optional[ForecastingHorizon]=None, coverage: float=0.9, refit: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]: Update cutoff value to forecast new dates.
    - get_pred_errors() -> pd.DataFrame: Get the prediction errors.

    Raises:
    -------
    - AssertionError: If the provided data is not a DataFrame, or if depvar_str is not a valid column in the data, or if exog_l is not None or an iterable.
    - AssertionError: If any column in exog_l is not present in the data.
    - AssertionError: If freq is not a recognized pandas frequency.
    """

    # Initializer
    def __init__(self,
                 data: pd.DataFrame,
                 depvar_str: str,
                 fh: int,
                 pct_initial_window: float,
                 step_length: int,
                 forecaster_name: Optional[str] = 'Naive',
                 forecaster: Optional[object] = None,
                 exog_l: Optional[list] = None,
                 freq: Optional[str] = 'D',
                 ) -> None:
        """Initializes the Forecast class with the provided parameters."""
        self.__init_test(data, depvar_str, exog_l, freq)
        self._y, self._X = self.__clean_data(data=data, 
                                           depvar_str=depvar_str,
                                           exog_l=exog_l,
                                           freq=freq)
        # Forecasting Model Parameters
        self.forecaster_name = forecaster_name

        if forecaster is None:
            # Name of the forecaster should be known only if the forecaster is null
            _err_msg = f'unsupported forecaster! Known forcasters are {list(CommonForecastingModels.keys())}'
            assert forecaster_name in CommonForecastingModels.keys(), _err_msg
            self.forecaster = CommonForecastingModels[forecaster_name]
        else:
            self.forecaster = forecaster

        # Forecasting parameters
        #1. horizon
        self._fh = ForecastingHorizon(np.arange(1, fh+1))
        #2. Initial training window
        self._initial_window = int(len(self._y)*pct_initial_window)
        #3. Step size
        self._step_length = step_length

        #4. Declare the cross validation window
        self._cv = ExpandingWindowSplitter(initial_window=self._initial_window,
                                          step_length=self._step_length,
                                          fh=self._fh)

        # Create the train test sets
        if self._X is None:
            self._X_train =  None
            self._X_test = None
            self._y_train, self._y_test = temporal_train_test_split(y=self._y, 
                                                                  train_size=self._initial_window)
        else:
            self._y_train, self._y_test, self._X_train, self._X_test = temporal_train_test_split(
            y=self._y, 
            X=self._X,
            train_size=self._initial_window)

        self.is_fitted = False
        self._fitted = None
        self.is_evaluated = False
        self._eval = None

        # Plots
        self.plot = self.__plot()

    def split_procedure_summary(self, verbose: bool=True) -> dict:
        """
        Generate a summary of the cross-validation procedure.

        Parameters:
        -----------        
            verbose : bool, optional
                If True, print the summary. Default is True.

        Returns:
        --------        
            dict
                A dictionary containing the summary of the cross-validation procedure.
        """

        _n_splits = self._cv.get_n_splits(self._y)
        cutoffs = [self._y.index[_train[-1]] for (_train, _) in self._cv.split(self._y.index)]
        _split_proc= {'Number of Folds': _n_splits,
                      'Initial Window Size': self._cv.initial_window,
                      'Step Length': self._cv.step_length,
                      'Forecast Horizon': len(self._cv.fh),
                      'First Cutoff': cutoffs[0],
                      'Last Curoff': cutoffs[-1]
                     }
        if verbose:
            for k, v in _split_proc.items():
                print(f"{k:<21}: {v}")
        return _split_proc    

    def fit(self, 
            on: str = 'all', 
            fh: Optional[ForecastingHorizon] = None
           ): #-> ForecastFit
        """
        Fit the forecaster and compute insample results.

        Parameters:
        --------
            on : str, optional
                Either 'train' or 'all'. Sample on which the model is fitted. By default, it is fitted on the entire sample.
            fh : ForecastingHorizon, optional
                Forecast horizon.

        Returns:
        --------
            ForecastFit:
                An instance of the ForecastFit class containing the fitted model and insample performance metrics.
        """

        if fh is None:
            fh =  self._fh

        if on == 'all':
            self.forecaster.fit(y=self._y, X=self._X, fh=fh)
            self.is_fitted = True

        elif on=='train':
            self.forecaster.fit(y=self._y_train, X=self._X_train, fh=fh)
            self.is_fitted = False

        else: 
            on_values =['all', 'train']
            raise ValueError(f'argument takes 2 possible values {on_values}')
        self._fitted = ForecastFit(self)   
        return self._fitted

    def evaluate(self): #-> ForecastEval
        """
        Evaluate the forecaster out-of-sample.

        Returns:
        --------
            ForecastEval:
                An instance of the ForecastEval class containing the out-of-sample evaluation results.
        """

        self._eval = ForecastEval(self)  
        self.is_evaluated = True
        return self._eval

    def predict(self, 
                X: Optional[pd.DataFrame] = None, 
                fh: Optional[ForecastingHorizon] = None, 
                coverage: float = 0.9,
                verbose = False
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions (Average and confidence interval) using the fitted model.

        Parameters:
        -----------        
            X : pd.DataFrame, optional
                Exogenous variables for forecasting. Default is None
            fh : ForecastingHorizon, optional
                Forecast horizon. Default is None and takes the horizon defined at instantiation.
            coverage : float, optional
                The coverage of the confidence interval. Default is 0.9.

        Returns:
        --------        
            Tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing the predictions and the confidence intervals.
        """
        if self.is_fitted==False:
            if verbose:
                print(f"\n{self.forecaster_name} model not fitted yet, or fitted on a subset only")
                print("Fitting the model on the entire sample ...")
            self.fit(on='all', fh=fh)

        if fh is None:
            fh =  self._fh
        if X is None:
            X = self._X

        y_pred = self.forecaster.predict(X=X, fh=fh)
        try:
            y_pred_ints = self.forecaster.predict_interval(X=X, fh=fh, coverage=coverage)
        except Exception as e:
            if verbose:
                print(f"{self.forecaster_name} does not support prediction intervals")
                print(f"Error: {e}")
                print("Computing the prediction intervals based on historical errors distribution")
            
            historical_errors = self.get_pred_errors()
            y_pred_ints = calculate_prediction_interval(historical_errors, y_pred, coverage=coverage)

        return y_pred, y_pred_ints

    def update(self, 
               new_y: pd.Series,
               new_X: Optional[pd.DataFrame] = None,
               fh: Optional[ForecastingHorizon] = None, 
               coverage: float = 0.9, 
               refit: bool = False,
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update cutoff value to forecast new dates.
        Possibility to refit the model.

        Parameters:
        -----------        
            new_y : pd.Series
                The new dependent variable values.
            new_X : pd.DataFrame, optional
                The new exogenous variables. Default is None.
            fh : ForecastingHorizon, optional
                Forecast horizon. Default is None and takes the horizon defined at instantiation.
            coverage : float, optional
                The coverage of the confidence interval. Default is 0.9.
            refit : bool, optional
                If True, refit the model. Default is False.

        Returns:
        --------        
            Tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing the updated predictions and the updated confidence intervals.
        """
        new_y = new_y.resample(self.freq).last().ffill()
        if new_X is not None:
            new_X = new_X.resample(self.freq).last().ffill()
        self.forecaster.update(y=new_y, X=new_X, update_params=refit)
        self._y = new_y
        self._X = new_X
        y_pred, y_pred_ints = self.predict(X=new_X, fh=fh, coverage=coverage)
        return y_pred, y_pred_ints

    def get_pred_errors(self):
        """
        Get the prediction errors.
        Returns:
        --------        
            pd.DataFrame:
                A DataFrame containing the prediction errors.
        """
        if (self.is_evaluated is False) or (self._eval is None):
            self._eval = self.evaluate()        
        try:
            pred_errors = self._eval._oos_horizon_df[['cutoff', 'horizon', 'error']]
        except Exception:
            return None
        return pred_errors
        
    def __plot(self): #-> ForecastPlot
        """
        Create an instance of the ForecastPlot class for plotting utility.

        Returns:
        --------
            ForecastPlot:
                An instance of the ForecastPlot class.
        """
        return ForecastPlot(self)  
    def __init_test(self,
                    data: pd.DataFrame, 
                    depvar_str: str, 
                    exog_l: Optional[list], 
                    freq: str
                   ) -> None:
        """
        Perform checks for the provided data, dependent variable, exogenous variables, and frequency.

        Parameters:
        -----------        
            data : pd.DataFrame
                A DataFrame containing the input data for forecasting.
            depvar_str : str
                The column name representing the dependent variable for forecasting.
            exog_l : list or None
                List of exogenous variables for forecasting.
            freq : str
                The frequency of the time series data.

        Returns:
        --------
            None

        Raises:
        -------        
            AssertionError:
                If the provided data is not a DataFrame, or if depvar_str is not a valid column in the data,
                or if exog_l is not None or an iterable.
            AssertionError:
                If any column in exog_l is not present in the data.
            AssertionError:
                If freq is not a recognized pandas frequency.
        """

        assert isinstance(data, pd.DataFrame), 'the data should be a dataframe'

        assert depvar_str in data.columns, 'the dependent varible must be a column in the data'
        self.depvar = depvar_str

        cond1 = exog_l is None
        cond2 = isinstance(exog_l, Iterable)
        assert cond1 or cond2, 'exog_l is either None, meaning no X or an iterable object'
        if cond2:
            assert all([c in data.columns for c in exog_l]), 'not all exog variables are in the data'
            self.exog_l = exog_l
        else:
            self.exog_l = exog_l

        # Pandas frequencies
        pandas_frequency_dict = {
            "D": "daily",
            "B": "business days",
            "W": "weekly (end of week, default on Sunday)",
            "M": "monthly (end of month)",
            "Q": "quarterly (end of quarter)",
            "A": "annual (end of year)",
            "H": "hourly",
            "T": "minutely",
            "S": "secondly",
            "L": "millisecondly",
            "U": "microsecondly",
            "BQ": "business quarterly (business quarter-end)",
            "BA": "business annual (business year-end)",
            "BH": "business hourly (business hour)"
        }
        assert freq in pandas_frequency_dict.keys(), f'Not a pandas recognized frequency. List of pandas frequencies:\n{pandas_frequency_dict}'
        self.freq = freq

        return None

    def __clean_data(self, 
                     data: pd.DataFrame, 
                     depvar_str: str, 
                     exog_l: Optional[list],
                     freq: str
                    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Reformat the data taking into account the requested frequency.
        """

        _df = data.dropna().resample(freq).last().copy()
        _df.index.freq = freq
        # Declare and stage the variable to forecast
        y = _df[depvar_str]
        

        # List of Exogenous variables if any
        if exog_l is None:
            X = None
        else:
            assert all([c in _df.columns for c in exog_l]), 'not all columns are not in the data'
            X = _df[exog_l]
        return y, X

class ForecastPlot:
    """
    Plotting utility class for Forecast.

    Parameters:
    -----------    
    - LF (Forecast): An instance of the Forecast class

    Methods:
    --------
    - plot_train_test(labels: List[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: str = 'Train-Test sets', ax: Optional[plt.Axes] = None, figsize: Tuple[float, float] = (15, 6)) -> Tuple[plt.Figure, np.array]: Plot the dependent variable separating the train from the test windows.
    - plot_cv_procedure(ax: Optional[plt.Axes] = None, labels: List[str] = None, ylabel: str = "Window number", xlabel: str = "Time", title: str = "Cross Validation Procedure") -> Tuple[plt.Figure, np.array]: Plot the cross-validation procedure.
    - plot_prediction(y_pred: pd.Series, y_pred_ints: Optional[pd.DataFrame] = None, interval_label: str = 'CI', labels: List[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: str = 'Prediction', ax: Optional[plt.Axes] = None, figsize: Tuple[float, float] = (15, 6)) -> Tuple[plt.Figure, np.array]: Plot the forecast predictions and the confidence intervals.
    - plot_prediction_true(y_pred: pd.Series, y_pred_ints: Optional[pd.DataFrame] = None, interval_label: str = 'CI', labels: List[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: str = 'Prediction', ax: Optional[plt.Axes] = None, figsize: Tuple[float, float] = (15, 6)) -> Tuple[plt.Figure, np.array]: Plot the forecast predictions, true values, and the confidence intervals.
    """

    def __init__(self, LF: Forecast):
        self._y_train = LF._y_train
        self._y_test = LF._y_test
        self._y = LF._y
        self._cv = LF._cv
        return None

    def plot_train_test(self,
                        labels: List[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        title: str = 'Train-Test sets',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[float, float] = (15, 6)
                        ):
        """
        Plot the dependent variable separating the train from the test windows.

        Parameters:
        -----------
            labels : List[str], optional
                Labels for the plot. Default is ["y_train", "y_test"].
            xlabel : str, optional
                Label for the x-axis.
            ylabel : str, optional
                Label for the y-axis.
            title : str, optional
                The title of the plot. Default is 'Train-Test sets'.
            ax : plt.Axes, optional
                The Axes object for the plot.
            figsize : Tuple[float, float], optional
                The figure size. Default is (15, 6).

        Returns:
        --------        
            fig : plt.Figure
                The Figure object containing the plot.
            axes : np.array
                An array of Axes objects containing the plot.
        """
        if labels is None:
            labels = ["y_train", "y_test"]
        return plot_series(self._y_train, self._y_test, 
                           labels =labels, 
                           ax = ax, 
                           xlabel = xlabel,
                           ylabel = ylabel,
                           title = title,
                           figsize = figsize)

    def plot_cv_procedure(self,
                          ax: Optional[plt.Axes] = None,
                          labels: List[str] = None,
                          ylabel: str = "Window number",
                          xlabel: str = "Time",
                          title: str = "Cross Validation Procedure"
                          ):
        """
        Plot the cross-validation procedure.

        Parameters:
        -----------        
            ax : plt.Axes, optional
                The Axes object for the plot.
            labels : List[str], optional
                Labels for the plot. Default is ["Window", "Forecasting horizon"].
            ylabel : str, optional
                Label for the y-axis. Default is "Window number".
            xlabel : str, optional
                Label for the x-axis. Default is "Time".
            title : str, optional
                The title of the plot. Default is "Cross Validation Procedure".

        Returns:
        --------        
            fig : plt.Figure
                The Figure object containing the plot.
            axes : np.array
                An array of Axes objects containing the plot.
        """

        _train_windows, _test_windows = self._get_windows()
        if labels is None:
            labels = ["Window", "Forecasting horizon"]
            
        return plot_windows(self._y, 
                            _train_windows,
                            _test_windows,
                            ax = ax,
                            labels = labels,
                            xlabel = xlabel, 
                            ylabel = ylabel,
                            title = title)

    def plot_prediction(self,
                        y_pred: pd.Series,
                        y_pred_ints: Optional[pd.DataFrame] = None,
                        interval_label: str = 'CI',
                        labels: List[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        title: str = 'Prediction',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[float, float] = (15, 6)):
        """
        Plot the forecast predictions and the confidence intervals.

        Parameters:
        ----------        
            y_pred : pd.Series
                The predicted values.
            y_pred_ints : pd.DataFrame, optional
                The DataFrame containing the prediction intervals.
            interval_label : str, optional
                Label for the prediction interval. Default is 'prediction interval'.
            labels : List[str], optional
                Labels for the plot. Default is ["y_train", "y_pred"].
            xlabel : str, optional
                Label for the x-axis.
            ylabel : str, optional
                Label for the y-axis.
            title : str, optional
                The title of the plot. Default is 'Prediction'.
            ax : plt.Axes, optional
                The Axes object for the plot.
            figsize : Tuple[float, float], optional
                The figure size. Default is (15, 6).

        Returns:
        -------        
            fig : plt.Figure
                The Figure object containing the plot.
            axes : np.array
                An array of Axes objects containing the plot.
        """

        y = self._y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-3*len(y_pred):]	
        if labels is None:
            labels = ["y_train", "y_pred"]
        return plot_series(zoom_y_train, y_pred,
                           labels= labels,
                           pred_interval=y_pred_ints,	
                           interval_label = interval_label,
                           xlabel = xlabel,
                           ylabel = ylabel,
                           title = title, 
                           ax = ax,
                           figsize = figsize
                          )

    def plot_prediction_true(self,
                             y_pred: pd.Series,
                             y_pred_ints: Optional[pd.DataFrame] = None,
                             interval_label: str = 'CI',
                             labels: List[str] = None,
                             xlabel: Optional[str] = None,
                             ylabel: Optional[str] = None,
                             title: str = 'Prediction',
                             ax: Optional[plt.Axes] = None,
                             figsize: Tuple[float, float] = (15, 6)):
        """
        Plot the forecast predictions, true values, and the confidence intervals.

        Parameters:
        -----------        
            y_pred : pd.Series
                The predicted values.
            y_pred_ints : pd.DataFrame, optional
                The DataFrame containing the prediction intervals.
            interval_label : str, optional
                Label for the prediction interval. Default is 'prediction interval'.
            labels : List[str], optional
                Labels for the plot. Default is ["y_train", "y_true", "y_pred"].
            xlabel : str, optional
                Label for the x-axis.
            ylabel : str, optional
                Label for the y-axis.
            title : str, optional
                The title of the plot. Default is 'Prediction'.
            ax : plt.Axes, optional
                The Axes object for the plot.
            figsize : Tuple[float, float], optional
                The figure size. Default is (15, 6).

        Returns:
        --------        
            fig : plt.Figure
                The Figure object containing the plot.
            axes : np.array
                An array of Axes objects containing the plot.
        """

        y = self._y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-5*len(y_pred):]
        true_pred_idx = np.intersect1d(y.index, y_pred.index)
        err_msg = 'No overlap between true values and predicted values.\nIf you want to plot prediction alone use the function plot_prediction'
        assert len(true_pred_idx)>0, err_msg
        y_true = self._y[true_pred_idx]
        if labels is None:
            labels = ["y_train", "y_true", "y_pred"]

        return plot_series(zoom_y_train, y_true, y_pred,
                           pred_interval=y_pred_ints,
                           interval_label =interval_label,
                           labels=labels, 
                           xlabel = xlabel,
                           ylabel = ylabel,
                           title = title, 
                           ax = ax,
                           figsize = figsize)

    def _get_windows(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate windows for the cross-validation procedure.

        Returns:
        --------
            tuple:
                A tuple containing two lists: train_windows and test_windows.
        """
        _train_windows = []
        _test_windows = []
        _y_index = self._y.index 
        for _train, _test in self._cv.split(_y_index):
            _train_windows.append(_train)
            _test_windows.append(_test)
        return _train_windows, _test_windows 

##############################################################################
# Model Fit and Insample Performance
##############################################################################
def compute_predictions(params):
    forecaster, X, intrain, intest, verbose = params
    fh = ForecastingHorizon(intest.index, is_relative=False)
    try:
        in_pred = forecaster.predict(fh=fh, X=X).rename('y_pred').reset_index()
        in_pred['y_true'] = intest.values
        in_pred['error'] = in_pred['y_true'] - in_pred['y_pred']
        in_pred['error_pct'] = in_pred['error'].abs()/in_pred['y_true']
        in_pred.insert(0, 'horizon', np.arange(1, len(in_pred) + 1))
        in_pred.insert(0, 'cutoff', intrain.index[-1])        
    except Exception as e:
        if verbose:
            print(f"Error occured in {intrain.index[-1]}: {e}")
        in_pred = pd.DataFrame()
    return in_pred
class ForecastFit:
    """
    Class for fitting the forecaster and computing insample predictions.

    Parameters:
    -----------    
    - Forecast (Forecast): An instance of the Forecast class.

    Attributes:
    -----------
    - forecaster (object): The forecasting model object.
    - forecaster_name (str): The name of the forecasting model.
    - is_fitted (bool): A flag indicating if the model is fitted.
    - _y_train (pd.Series): The training set of the dependent variable.
    - _y (pd.Series): The dependent variable for forecasting.
    - _X (pd.DataFrame): The exogenous variables for forecasting.
    - _fh (ForecastingHorizon): The forecast horizon.
    - _cv (ExpandingWindowSplitter): The cross-validation window.
    - plot (ForecastFitPlot): An instance of the ForecastFitPlot class for plotting utility.
    - insample_result_df (pd.DataFrame): A DataFrame containing the insample predictions.
    - insample_perf_summary (dict): A dictionary containing the computed insample performance metrics.

    Methods:
    --------
    - insample_predictions(random_sample: bool=False, nsample: int=100, verbose: bool=False) -> pd.DataFrame: Compute the insample predictions for the fitted model.
    - insample_perf() -> dict: Compute insample performance metrics (RMSE and MAPE) for the fitted model.
    """

    def __init__(self, LF: Forecast):
        self.forecaster = LF.forecaster
        self.forecaster_name = LF.forecaster_name
        self.is_fitted = LF.is_fitted
        self._y_train = LF._y_train
        self._y = LF._y
        self._X = LF._X
        self._fh = LF._fh
        self._cv = LF._cv
        self.plot = self.__plot()

        self.insample_result_df = None
        self.insample_perf_summary = None

    def insample_predictions(self, random_sample=False, nsample: int = 100, verbose=False) -> pd.DataFrame:
        """
        Compute the insample predictions for the fitted model.

        Parameters:
        -----------        
            nsample : int, optional
                The number of samples to compute. Default is 100.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing the insample predictions.
        """

        if self.is_fitted:
            _y = self._y
        else:
            _y = self._y_train

        insample_eval_window = len(_y) - len(self._fh)
        if random_sample:            
            nsample = max(insample_eval_window, nsample)
            # Randomly selecting cutoff points
            _cutoffs = np.random.choice(insample_eval_window, size=nsample, replace=False)
            cv_in = CutoffSplitter(cutoffs=_cutoffs, window_length=1, fh=self._fh)
        else:
            _cutoffs = np.arange(insample_eval_window)
            cv_in = ExpandingWindowSplitter(initial_window=1, step_length=1, fh=self._fh)
        if verbose:
            print(f"\nComputing {self.forecaster_name} forecaster historic predictions....")        

        params = [(self.forecaster, self._X, intrain, intest, verbose) for intrain, intest in cv_in.split_series(_y)]                
        with Pool() as pool:
            insample_result = list(tqdm(pool.imap(compute_predictions, params), total=len(params)))

        insample_result_df = pd.concat(insample_result)
        if insample_result_df.empty:
            print(f"No insample predictions computed {self.forecaster_name}")
        if verbose:
            print(f"\n{self.forecaster_name} forecaster historic predictions completed")        
        self.insample_result_df = insample_result_df
        return insample_result_df

    def insample_perf(self) -> dict:
        """
        Compute insample performance metrics (RMSE and MAPE) for the fitted model.

        Returns:
        --------        
            dict
                A dictionary containing the computed insample performance metrics.
        """

        if self.insample_result_df is None:
            self.insample_predictions()

        insample_perf_summary = {
            'cutoff': summary_perf(self.insample_result_df, grouper = 'cutoff', y_true_col = 'y_true', y_pred_col = 'y_pred'),
            'horizon':summary_perf(self.insample_result_df, grouper = 'horizon', y_true_col = 'y_true', y_pred_col = 'y_pred')
        }
        self.insample_perf_summary = insample_perf_summary
        return insample_perf_summary

    def __plot(self):
        return ForecastFitPlot(self)

class ForecastFitPlot:
    """
    Plotting utility class for ForecastFit.

    Parameters:
    -----------    
    - LFF (ForecastFit): An instance of the ForecastFit class.

    Methods:
    --------
    - plot_insample_performance(metric: str = 'RMSE', title: str = 'Insample Performance') -> Tuple[plt.Figure, np.array]: Plot the insample performance metrics.
    """

    def __init__(self, LFF: ForecastFit):
        self._LFF = LFF

    def plot_insample_performance(self,
                                  metric: str = 'RMSE',
                                  title: str = 'Insample Performance'):
        """
        Plot the insample performance metrics.

        Parameters:
        -----------        
            metric : str, optional
                The performance metric to plot. Default is 'RMSE'.
            title : str, optional
                The title of the plot. Default is 'Insample Performance'.

        Returns:
        --------        
            fig : plt.Figure
                The Figure object containing the plot.
            axes : np.array
                An array of Axes objects containing the plot.
        """

        assert metric in ['RMSE', 'MAPE'], f'{metric} not in summary performance'
        if 'insample_perf_summary' not in self.__dict__.keys():
            insample_perf_summary = self._LFF.insample_perf()

        f, axes = plt.subplots(1,2,figsize=(15,5))
        for i, (_grouper, _df) in enumerate(insample_perf_summary.items()):
            _df[metric].plot(ax= axes[i], style = '-o', title = f'{metric} By {_grouper}')
            axes[i].set_xlabel('')

        plt.suptitle(title)
        plt.tight_layout()                   
        return f, axes

##############################################################################
# Model Out of Sample Evaluation
##############################################################################
class ForecastEval:
    """
    Class for evaluating the forecaster out-of-sample.

    Parameters:
    -----------    
    - Forecast (Forecast): An instance of the Forecast class.

    Attributes:
    -----------
    - forecaster (object): The forecasting model object.
    - forecaster_name (str): The name of the forecasting model.
    - oos_eval (pd.DataFrame): A DataFrame containing the out-of-sample evaluation results.
    - plot (ForecastEvalPlot): An instance of the ForecastEvalPlot class for plotting utility.
    - _y (pd.Series): The dependent variable for forecasting.
    - _X (pd.DataFrame): The exogenous variables for forecasting.
    - _cv (ExpandingWindowSplitter): The cross-validation window.
    - _scoring_metrics (list): A list of scoring metrics.    
    - _oos_horizon_df (pd.DataFrame): A DataFrame containing the out-of-sample predictions and errors per horizon.
    - _oos_horizon_perf (pd.DataFrame): A DataFrame containing the summary performance metrics per horizon.
    - _oos_cutoff_perf (pd.DataFrame): A DataFrame containing the summary performance metrics per cutoff.
    

    Methods:
    --------
    - summary_results() -> pd.DataFrame: Generate a summary of out-of-sample forecast results.
    - summary_cutoff() -> pd.DataFrame: Generate a summary of out-of-sample performance per cutoff.
    - summary_horizon() -> pd.DataFrame: Generate a summary of out-of-sample performance per horizon.
    """

    def __init__(self, LF: Forecast):
        
        self.forecaster = LF.forecaster
        self.forecaster_name = LF.forecaster_name
        self._y = LF._y
        self._X = LF._X
        self._cv = LF._cv

        self._scoring_metrics = [MeanSquaredError(square_root=True),
                                MeanAbsoluteError(),
                                MeanAbsolutePercentageError(), 
                                MedianAbsoluteError(),
                                ]

        _rename_metrics = {
            'test_MeanSquaredError':'RMSE', 
            'test_MeanAbsoluteError':'MAE',
            'test_MeanAbsolutePercentageError':'MAPE',
            'test_MedianAbsoluteError':'MedianAE',
        }
        print(f"\nStart {self.forecaster_name} forecaster evalution....")
        st = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.oos_eval = evaluate(forecaster=self.forecaster, 
                                y=self._y,
                                X = self._X,
                                cv=self._cv,
                                strategy="refit",
                                return_data=True,
                                scoring = self._scoring_metrics,
                                backend ='loky',
                            )
        self.oos_eval = self.oos_eval.set_index('cutoff').sort_index()
        self.oos_eval = self.oos_eval.rename(columns = _rename_metrics)
        et = time.time()
        elapsed_time = et - st
        print(f"Evaluation completed in: {np.around(elapsed_time / 60,3)} minutes")

        convert_horizon = self.oos_eval.apply(self.__eval_horizon, axis=1)
        
        self._oos_horizon_df = pd.concat(convert_horizon.values)
        self._oos_horizon_perf = summary_perf(self._oos_horizon_df, 
                                             grouper='horizon', 
                                             y_true_col = 'y_test', 
                                             y_pred_col = 'y_pred')
        self._oos_cutoff_perf = summary_perf(self._oos_horizon_df, 
                                             grouper='cutoff', 
                                             y_true_col = 'y_test', 
                                             y_pred_col = 'y_pred')

        self.plot = self.__plot()
        return None

    def summary_results(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample forecast results.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing various summary statistics of the out-of-sample forecasts.
        """

        _summary = {
         'Number of Folds':  self.oos_eval.shape[0], 
         'Avg Fit time (s)': self.oos_eval.fit_time.mean(),
         'Avg_pred_time (s)': self.oos_eval.pred_time.mean(),
         'Smallest training window': self.oos_eval.len_train_window.min(),
         'Largest training window':self.oos_eval.len_train_window.max(),
         'First cutoff': self.oos_eval.index[0],
         'Last cutoff': self.oos_eval.index[-1],
        }
        for _s in self._oos_cutoff_perf.columns:
            _summary[f'Avg {_s}'] = self._oos_cutoff_perf[_s].mean()
        return pd.Series(_summary).to_frame().T

    def summary_cutoff(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample performance per cutoff.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing summary performance metrics (RMSE and MAPE) for each horizon.
        """        
        return self._oos_cutoff_perf

    def summary_horizon(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample performance per horizon.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing summary performance metrics (RMSE and MAPE) for each horizon.
        """        
        return self._oos_horizon_perf

    def __eval_horizon(self, x):
        _fct = pd.concat([x['y_test'], x['y_pred']], keys=['y_test', 'y_pred'], axis=1)
        _fct['error'] = _fct['y_test'] - _fct['y_pred']
        _fct['error_pct'] = _fct['error'].abs()/ _fct['y_test']
        _fct['horizon'] = np.arange(1, len(_fct)+1)    
        _fct['cutoff'] = x.name
        return _fct

    def __plot(self):
        return ForecastEvalPlot(self)

class ForecastEvalPlot:
    """
    Plotting utility class for ForecastEval.

    Parameters:
    -----------    
    LFE (ForecastEval): An instance of the ForecastEval class.

    Attributes:
    -----------
    _oos_horizon_perf (pd.DataFrame): A DataFrame containing the summary performance metrics per horizon.
    _oos_cutoff_perf (pd.DataFrame): A DataFrame containing the summary performance metrics per cutoff.

    Methods:
    --------
    plot_oos_score(score: str = 'RMSE', view: str = 'horizon', xlabel: str = None, ylabel: str = None, title: str = 'Out of Sample Performance', ax: Optional[plt.Axes] = None, figsize: Tuple[float, float] = (15, 6)) -> Tuple[plt.Figure, np.array]: Plot out-of-sample performance metric historically.
    """

    def __init__(self, LFE: ForecastEval):
        self._oos_horizon_perf = LFE._oos_horizon_perf
        self._oos_cutoff_perf = LFE._oos_cutoff_perf
        return None

    def plot_oos_score(self,
                       score: str = 'RMSE',
                       view: str = 'horizon', 
                       xlabel: str = None,
                       ylabel: str = None,
                       title: str = 'Out of Sample Performance',
                       ax: Optional[plt.Axes] = None,
                       figsize: Tuple[float, float] = (15, 6)):
        """
        Plot out-of-sample performance metric historically.

        Parameters:
        -----------        
            score : str, optional
                The performance metric to plot. Default is 'RMSE'.
            view: str, optional
                The view of the plot. It must be either 
                    - horizon: the function plots the average score per forecast horizon
                    - cutoff: the function plots the average across all horizons for each cutoff                
                Default is 'horizon'.
            xlabel : str, optional
                Label for the x-axis.
            ylabel : str, optional
                Label for the y-axis.
            title : str, optional
                The title of the plot. Default is 'Out of Sample Performance'.
            ax : plt.Axes, optional
                The Axes object for the plot.
            figsize : Tuple[float, float], optional
                The figure size. Default is (15, 6).
        
        Returns:
        --------        
            fig : plt.Figure
                The Figure object containing the plot.
            axes : np.array
                An array of Axes objects containing the plot.

        """
        if view == 'horizon':
            assert score in self._oos_horizon_perf.columns, 'score not computed'
            to_plot = self._oos_horizon_perf[score]
        elif view == 'cutoff':
            assert score in self._oos_cutoff_perf.columns, 'score not computed'
            to_plot = self._oos_cutoff_perf[score]
        else:
            raise ValueError('view should be either horizon or cutoff')

        if ax is None:
            f, ax = plt.subplots(1,1,figsize=figsize)

        ylabel = score if ylabel is None else ylabel
        xlabel = '' if xlabel is None else xlabel        
        to_plot.plot(ax = ax, style = '-o')                    
        ax.set(xlabel = xlabel, ylabel=ylabel)
        ax.set_title(title, size ='xx-large')
        if ax is None:
            return f, ax
        else:
            return ax