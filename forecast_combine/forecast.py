""" Timeseries Forecasting with Insample, Validation and Out-of-Sample predictions"""
__description__ = "Time Series Forecast"
__author__ = "Amine Raboun - amineraboun@github.io"

##############################################################################
# Import Libraries & default configuration 
##############################################################################
# Standard Inports
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

# Iteration tools
from collections.abc import Iterable
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
from sktime.split import CutoffSplitter
from sktime.forecasting.model_selection import (
ExpandingWindowSplitter, 
temporal_train_test_split
)
from sktime.forecasting.model_evaluation import evaluate
# Performance Metrics
from forecast_combine.utils.metrics import summary_perf
from sktime.performance_metrics.forecasting import (
MeanSquaredError,
MeanAbsoluteError, 
MeanAbsolutePercentageError, 
MedianAbsoluteError
)

# Filter Warnings
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

##############################################################################
# Default values
##############################################################################
# Common Forcasting models
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.statsforecast import (
StatsForecastAutoARIMA,
StatsForecastAutoETS, 
StatsForecastAutoTheta,
StatsForecastAutoTBATS, 
StatsForecastMSTL
)
CommonForecastingModels = {
"Naive": NaiveForecaster(),
"AutoARIMA": StatsForecastAutoARIMA(),
"AutoETS": StatsForecastAutoETS(),
"AutoTheta": StatsForecastAutoTheta(),
"TBATS": StatsForecastAutoTBATS(seasonal_periods = 1),
"LOESS": StatsForecastMSTL(season_length=1),
"Prophet": Prophet(),
}
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

##############################################################################
# Master Class Definition and Plot
##############################################################################
class Forecast(object):
    """
    Forecast class for managing and evaluating forecast models.

    Parameters
    
        data : pd.DataFrame
            A DataFrame containing the input data for forecasting.
        depvar_str : str
            The column name representing the dependent variable for forecasting.
        fh : int
            The forecast horizon, i.e., the number of periods ahead to forecast.
        pct_initial_window : float
            The percentage of data used as the initial training window.
        step_length : int
            The step size for expanding window cross-validation.
        forecaster_name : str, optional
            The name of the forecasting model. Default is 'Naive'.
        forecaster : object, optional
            The forecasting model object. Default is None, which will use the model corresponding to forecaster_name.
        exog_l : list, optional
            List of exogenous variables for forecasting. Default is None.
        freq : str, optional
            The frequency of the time series data. Default is 'B' (business days).
        random_state : int, optional
            Random state for reproducibility. Default is 0.

    Attributes
    
        y : pd.Series
            The time series data representing the dependent variable.
        X : pd.DataFrame or None
            The DataFrame containing exogenous variables or None if there are no exogenous variables.
        forecaster_name : str
            The name of the forecasting model used.
        forecaster : object
            The forecasting model object used for forecasting.
        fh : ForecastingHorizon
            The forecast horizon, i.e., the number of periods ahead to forecast.
        initial_window : int
            The size of the initial training window.
        step_length : int
            The step size for expanding window cross-validation.
        cv : ExpandingWindowSplitter
            The cross-validation window used for expanding window validation.
        X_train : pd.DataFrame or None
            The DataFrame containing exogenous variables for the training set or None if there are no exogenous variables.
        X_test : pd.DataFrame or None
            The DataFrame containing exogenous variables for the test set or None if there are no exogenous variables.
        y_train : pd.Series
            The dependent variable values for the training set.
        y_test : pd.Series
            The dependent variable values for the test set.
        is_fitted : bool
            True if the forecaster is fitted, False otherwise.
        is_evaluated : bool
            True if the forecaster is evaluated on the test set, False otherwise.
        rs : numpy.random.RandomState
            Random state for reproducibility.
        plot : ForecastPlot
            An instance of the ForecastPlot class for plotting utility.
    """

    # Initializer
    def __init__(self,
                 data: pd.DataFrame,
                 depvar_str: str,
                 fh: int,
                 pct_initial_window: float,
                 step_length: int,
                 forecaster_name: str = 'Naive',
                 forecaster: Optional[object] = None,
                 exog_l: Optional[list] = None,
                 freq: str = 'D',
                 ) -> None:
        
        self.__init_test(data, depvar_str, exog_l, freq)
        self.y, self.X = self.__clean_data(data=data, 
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
        self.fh = ForecastingHorizon(np.arange(1, fh+1))
        #2. Initial training window
        self.initial_window = int(len(self.y)*pct_initial_window)
        #3. Step size
        self.step_length = step_length

        #4. Declare the cross validation window
        self.cv = ExpandingWindowSplitter(initial_window=self.initial_window,
                                          step_length=self.step_length,
                                          fh=self.fh)

        # Create the train test sets
        if self.X is None:
            self.X_train =  None
            self.X_test = None
            self.y_train, self.y_test = temporal_train_test_split(y=self.y, 
                                                                  train_size=self.initial_window)
        else:
            self.y_train, self.y_test, self.X_train, self.X_test = temporal_train_test_split(
            y=self.y, 
            X=self.X,
            train_size=self.initial_window)

        self.is_fitted = False
        self.fitted = None
        self.is_evaluated = False
        self.eval = None

        # Plots
        self.plot = self.__plot()

    def split_procedure_summary(self, verbose=True) -> dict:
        """
        Generate a summary of the cross-validation procedure.

        Returns:
        --------
            dict:
                A dictionary containing details of the cross-validation procedure, including the number of folds, initial window size, step length, and forecast period.
        """

        _n_splits = self.cv.get_n_splits(self.y)
        cutoffs = [self.y.index[_train[-1]] for (_train, _) in self.cv.split(self.y.index)]
        _split_proc= {'Number of Folds': _n_splits,
                      'Initial Window Size': self.cv.initial_window,
                      'Step Length': self.cv.step_length,
                      'Forecast Horizon': len(self.cv.fh),
                      'First Cutoff': cutoffs[0],
                      'Last Curoff': cutoffs[-1]
                     }
        if verbose:
            for k, v in _split_proc.items():
                print(f"{k:<21}: {v}")
        return _split_proc    

    def __plot(self): #-> ForecastPlot
        """
        Create an instance of the ForecastPlot class for plotting utility.

        Returns:
        --------
            ForecastPlot:
                An instance of the ForecastPlot class.
        """
        return ForecastPlot(self)  

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
            fh =  self.fh

        if on == 'all':
            self.forecaster.fit(y=self.y, X=self.X, fh=fh)
            self.is_fitted = True

        elif on=='train':
            self.forecaster.fit(y=self.y_train, X=self.X_train, fh=fh)
            self.is_fitted = False

        else: 
            on_values =['all', 'train']
            raise ValueError(f'argument takes 2 possible values {on_values}')
        self.fitted = ForecastFit(self)   
        return self.fitted

    def evaluate(self): #-> ForecastEval
        """
        Evaluate the forecaster out-of-sample.

        Returns:
        --------
            ForecastEval:
                An instance of the ForecastEval class containing the out-of-sample evaluation results.
        """

        self.eval = ForecastEval(self)  
        self.is_evaluated = True
        return self.eval

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
                Exogenous variables for forecasting. Default is None and takes the exogenous variables defined at instantiation.
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
            fh =  self.fh
        if X is None:
            X = self.X

        y_pred = self.forecaster.predict(X=X, fh=fh)
        try:
            y_pred_ints = self.forecaster.predict_interval(X=X, fh=fh, coverage=coverage)
        except Exception as e:
            print(f"{self.forecaster_name} does not support prediction intervals")
            print(f"Error: {e}")
            y_pred_ints = None

        return y_pred, y_pred_ints

    def update(self, 
               newdata: pd.DataFrame, 
               fh: Optional[ForecastingHorizon] = None, 
               coverage: float = 0.9, 
               refit: bool = False,
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update cutoff value to forecast new dates.
        Possibility to refit the model.

        Parameters:
        -----------        
            newdata : pd.DataFrame
                The new data containing the same columns as the original data.
            fh : ForecastingHorizon, optional
                Forecast horizon. Default is None and takes the horizon defined at instantiation.
            coverage : float, optional
                The coverage of the confidence interval. Default is 0.9.
            refit : bool, optional
                If True, the model will be refitted on the new training data. Default is False.

        Returns:
        --------        
            Tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing the updated predictions and the updated confidence intervals.
        """

        new_y, new_X = self.__clean_data(data=newdata, 
                                           depvar_str=self.depvar,
                                           exog_l = self.exog_l,
                                           freq = self.freq)

        self.forecaster.update(y=new_y, X=new_X, update_params=refit)
        y_pred, y_pred_ints = self.predict(X=new_X, fh=fh, coverage=coverage)
        return y_pred, y_pred_ints

    def get_pred_errors(self):
        """
        Get the prediction errors.
        """
        if self.fitted is None:
            self.fit(on='all')
        if self.fitted.insample_result_df is None:
            self.fitted.insample_predictions()
        return self.fitted.insample_result_df[['cutoff', 'horizon', 'error']]

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
            assert all([c in data.columns for c in exog_l]), 'not all columns are not in the data'
            self.exog_l = exog_l
        else:
            self.exog_l = exog_l

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

        # Declare and stage the variable to forecast
        y = data[depvar_str].dropna().resample(freq).last().ffill()

        # List of Exogenous variables if any
        if exog_l is None:
            X = None
        else:
            assert all([c in data.columns for c in exog_l]), 'not all columns are not in the data'
            X = data[exog_l].resample(freq).last().ffill()
            X = X.loc[y.index]

        return y, X

class ForecastPlot:
    """
    Plotting utility class for Forecast.

    Parameters:
    -----------    
        LF : Forecast
            An instance of the Forecast class
    """

    def __init__(self, LF: Forecast):
        self.y_train = LF.y_train
        self.y_test = LF.y_test
        self.y = LF.y
        self.cv = LF.cv
        return None

    def plot_train_test(self,
                        labels: List[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        title: str = 'Train-Test sets',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[float, float] = (15, 6)) -> None:
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
            None
        """
        if labels is None:
            labels = ["y_train", "y_test"]
        return plot_series(self.y_train, self.y_test, 
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
                          ) -> None:
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
            None
        """

        _train_windows, _test_windows = self._get_windows()
        if labels is None:
            labels = ["Window", "Forecasting horizon"]
            
        return plot_windows(self.y, 
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
            None
        """

        y = self.y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-5*len(y_pred):]	
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
            None
        """

        y = self.y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-5*len(y_pred):]
        true_pred_idx = np.intersect1d(y.index, y_pred.index)
        err_msg = 'No overlap between true values and predicted values.\nIf you want to plot prediction alone use the function plot_prediction'
        assert len(true_pred_idx)>0, err_msg
        y_true = self.y[true_pred_idx]
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
        _y_index = self.y.index 
        for _train, _test in self.cv.split(_y_index):
            _train_windows.append(_train)
            _test_windows.append(_test)
        return _train_windows, _test_windows 

##############################################################################
# Model Fit and Insample Performance
##############################################################################

class ForecastFit:
    """
    Class for fitting the forecast model and computing insample performance metrics.

    Parameters:
    -----------    
        Forecast : Forecast
            An instance of the Forecast class.
    """

    def __init__(self, LF: Forecast):
        self.forecaster = LF.forecaster
        self.forecaster_name = LF.forecaster_name
        self.is_fitted = LF.is_fitted
        self.y_train = LF.y_train
        self.y = LF.y
        self.X = LF.X
        self.fh = LF.fh
        self.cv = LF.cv
        self.plot = self.__plot()

        self.insample_result_df = None
        self.insample_perf_summary = None

    def insample_predictions(self, random_sample=False, nsample: int = 100) -> pd.DataFrame:
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

        def __compute_predictions(forecaster, X, intrain, intest):
            fh = ForecastingHorizon(intest.index, is_relative=False)
            in_pred = forecaster.predict(fh=fh, X=X).rename('y_pred').reset_index()
            in_pred['y_true'] = intest.values
            in_pred['error'] = in_pred['y_true'] - in_pred['y_pred']
            in_pred['error_pct'] = in_pred['error'].abs()/in_pred['y_true']
            in_pred.insert(0, 'horizon', np.arange(1, len(in_pred) + 1))
            in_pred.insert(0, 'cutoff', intrain.index[-1])        
            return in_pred
        
        if self.is_fitted:
            _y = self.y
        else:
            _y = self.y_train

        insample_eval_window = len(_y) - len(self.fh)
        if random_sample:            
            nsample = max(insample_eval_window, nsample)
            # Randomly selecting cutoff points
            _cutoffs = np.random.choice(insample_eval_window, size=nsample, replace=False)
        else:
            _cutoffs = np.arange(insample_eval_window)

        print(f"\nComputing {self.forecaster_name} forecaster historic predictions....")
        cv_in = CutoffSplitter(cutoffs=_cutoffs, window_length=1, fh=self.fh)
        insample_result = [__compute_predictions(self.forecaster, self.X, intrain, intest) for intrain, intest in tqdm(cv_in.split_series(_y))]

        insample_result_df = pd.concat(insample_result)
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
            'horizon': summary_perf(self.insample_result_df, grouper = 'horizon', y_true_col = 'y_true', y_pred_col = 'y_pred')
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
        LFF : ForecastFit
            An instance of the ForecastFit class.
    """

    def __init__(self, LFF: ForecastFit):

        self.__dict__.update(LFF.__dict__)
        self.LFF = LFF

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
            insample_perf_summary = self.LFF.insample_perf()

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
        Forecast : Forecast
            An instance of the Forecast class.
    """

    def __init__(self, LF: Forecast):
        
        self.forecaster = LF.forecaster
        self.forecaster_name = LF.forecaster_name
        self.y = LF.y
        self.X = LF.X
        self.cv = LF.cv

        self.scoring_metrics = [MeanSquaredError(square_root=True),
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
        self.oos_eval = evaluate(forecaster=self.forecaster, 
                            y=self.y,
                            X = self.X,
                            cv=self.cv,
                            strategy="refit",
                            return_data=True,
                            scoring = self.scoring_metrics,
                            backend ='loky',
                           )
        self.oos_eval = self.oos_eval.set_index('cutoff').sort_index()
        self.oos_eval = self.oos_eval.rename(columns = _rename_metrics)
        et = time.time()
        elapsed_time = et - st
        print(f"Evaluation completed in: {np.around(elapsed_time / 60,3)} minutes")

        convert_horizon = self.oos_eval.apply(self.__eval_horizon, axis=1)
        
        self.oos_horizon_df = pd.concat(convert_horizon.values)
        self.oos_horizon_perf = summary_perf(self.oos_horizon_df, 
                                             grouper='horizon', 
                                             y_true_col = 'y_test', 
                                             y_pred_col = 'y_pred')
        self.oos_cutoff_perf = summary_perf(self.oos_horizon_df, 
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
        for _s in self.oos_cutoff_perf.columns:
            _summary[f'Avg {_s}'] = self.oos_cutoff_perf[_s].mean()
        return pd.Series(_summary).to_frame().T

    def summary_cutoff(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample performance per cutoff.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing summary performance metrics (RMSE and MAPE) for each horizon.
        """        
        return self.oos_cutoff_perf

    def summary_horizon(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample performance per horizon.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing summary performance metrics (RMSE and MAPE) for each horizon.
        """        
        return self.oos_horizon_perf

    def __eval_horizon(self, x):
        _fct = pd.concat([x['y_test'], x['y_pred']], keys=['y_test', 'y_pred'], axis=1)
        _fct['error_pct'] = (_fct['y_test'] - _fct['y_pred']).abs()/ _fct['y_test']
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
        LFE : ForecastEval
            An instance of the ForecastEval class.
    """

    def __init__(self, LFE: ForecastEval):
        self.oos_horizon_perf = LFE.oos_horizon_perf
        self.oos_cutoff_perf = LFE.oos_cutoff_perf
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
            assert score in self.oos_horizon_perf.columns, 'score not computed'
            to_plot = self.oos_horizon_perf[score]
        elif view == 'cutoff':
            assert score in self.oos_cutoff_perf.columns, 'score not computed'
            to_plot = self.oos_cutoff_perf[score]
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