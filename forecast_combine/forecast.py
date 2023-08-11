# Standard Inports
import pandas as pd
import numpy as np
import os

from typing import Optional, Tuple, List
from collections.abc import Iterable
from tqdm.autonotebook import tqdm
import time

# Plotting packages
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set(style='white', font_scale=1)
plt.rc('axes', titlesize='large')    
plt.rc('axes', labelsize='large')   
plt.rc('xtick', labelsize='large')   
plt.rc('ytick', labelsize='large')   
plt.rc('legend', fontsize='large')   
plt.rc('figure', titlesize='x-large') 
from forecast_combine.utils.plotting import plot_series

# Forcast models
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import (
StatsForecastAutoARIMA,
StatsForecastAutoETS, 
StatsForecastAutoTheta
)
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.fbprophet import Prophet

ForecastingModels = {
"Naive": NaiveForecaster(),
"AutoARIMA": StatsForecastAutoARIMA(),
"AutoETS": StatsForecastAutoETS(),
"AutoTheta": StatsForecastAutoTheta(),
"TBATS": TBATS(),
"Prophet": Prophet(),
}

# Cross Validation
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
CutoffSplitter, 
ExpandingWindowSplitter, 
temporal_train_test_split
)
from sktime.forecasting.model_evaluation import evaluate
# Performance Metrics
from sktime.performance_metrics.forecasting import (
MeanAbsolutePercentageError, 
MeanSquaredError,
)

# Filter Warnings
from warnings import simplefilter
simplefilter('ignore')

def summary_perf(insample_result_df, 
			 grouper, 
			 y_true_col = 'y_true',
			 y_pred_col = 'y_pred',
			):

	"""
	Compute summary performance metrics for a given forecast.

	Parameters:
		- insample_result_df (DataFrame): A DataFrame containing the forecast results.
		- grouper (str): The column name to group the forecast results for computing summary metrics.
		- y_true_col (str, optional): The column name representing the true values. Default is 'y_true'.
		- y_pred_col (str, optional): The column name representing the predicted values. Default is 'y_pred'.

	Returns:
		DataFrame: A DataFrame containing summary performance metrics, including RMSE and MAPE.
	"""

	rmse = MeanSquaredError(square_root=True)
	mape = MeanAbsolutePercentageError()
	horizon_metrics = pd.concat([insample_result_df.groupby(grouper).apply(lambda x: rmse(x[y_true_col], x[y_pred_col])).rename('RMSE'),
									 insample_result_df.groupby(grouper).apply(lambda x: mape(x[y_true_col], x[y_pred_col])).rename('MAPE')
									], axis=1)
	return horizon_metrics

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

    __description = "Time Series Forecast"
    __author = "Amine Raboun - amineraboun@github.io"

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
                 freq: str = 'B',
                 random_state: int = 0) -> None:
        """
        Initialize the Forecast instance.

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
        """
        
        self.__init_test(data, depvar_str, exog_l, freq)
        self.y, self.X = self.__clean_data(data=data, 
                                       depvar_str=depvar_str,
                                       exog_l=exog_l,
                                       freq=freq)
        # Forecasting Model Parameters
        self.forecaster_name = forecaster_name

        if forecaster is None:
            # Name of the forecaster should be known only if the forecaster is null
            _err_msg = f'unsupported forecaster! Known forcasters are {list(ForecastingModels.keys())}'
            assert forecaster_name in ForecastingModels.keys(), _err_msg
            self.forecaster = ForecastingModels[forecaster_name]
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
            self.y_train, self.y_test = temporal_train_test_split(
                y=self.y,
                train_size=self.initial_window)
        else:
            self.y_train, self.y_test, self.X_train, self.X_test = temporal_train_test_split(
            y=self.y, 
            X=self.X,
            train_size=self.initial_window)

        self.is_fitted = False
        self.is_evaluated = False
        self.rs = np.random.RandomState(random_state) 
        # Plots
        self.plot = self.__plot()


    def split_procedure_summary(self) -> dict:
        """
        Generate a summary of the cross-validation procedure.

        Returns
        
            dict:
                A dictionary containing details of the cross-validation procedure, including the number of folds, initial window size, step length, and forecast period.
        """

        _n_splits = cv.get_n_splits(self.y)
        cutoffs = [_train[0] for (_train, _test) in self.cv.split(self.y.index)]
        _split_proc= {'Number of Folds': _n_splits,
                      'Initial Window Size': cv.initial_window,
                      'Step Length': cv.step_length,
                      'Forecast period': len(cv.fh),
                      'First Cutoff': cutoffs[0],
                      'Last Curoff': cutoffs[-1]
                     }
        return _split_proc    

    def __plot(self): #-> ForecastPlot
        """
        Create an instance of the ForecastPlot class for plotting utility.

        Returns:
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
            on : str, optional
                Either 'train' or 'all'. Sample on which the model is fitted. By default, it is fitted on the entire sample.
            fh : ForecastingHorizon, optional
                Forecast horizon.

        Returns:
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

        return ForecastFit(self)  

    def evaluate(self): #-> ForecastEval
        """
        Evaluate the forecaster out-of-sample.

        Returns:
            ForecastEval:
                An instance of the ForecastEval class containing the out-of-sample evaluation results.
        """

        self.eval = ForecastEval(self)  
        self.is_evaluated = True
        return self.eval

    def predict(self, 
                X: Optional[pd.DataFrame] = None, 
                fh: Optional[ForecastingHorizon] = None, 
                coverage: float = 0.9
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions (Average and confidence interval) using the fitted model.

        Parameters
        
            X : pd.DataFrame, optional
                Exogenous variables for forecasting. Default is None and takes the exogenous variables defined at instantiation.
            fh : ForecastingHorizon, optional
                Forecast horizon. Default is None and takes the horizon defined at instantiation.
            coverage : float, optional
                The coverage of the confidence interval. Default is 0.9.

        Returns
        
            Tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing the predictions and the confidence intervals.
        """
        if self.is_fitted==False:
            print("\nModel not fitted yet, or fitted on training sample alone")
            print("Fitting the model on the whole sample ...")
            self.fit(on='all', fh=fh)

        if fh is None:
            fh =  self.fh
        if X is None:
            X = self.X

        y_pred = self.forecaster.predict(X=X, fh=fh)
        y_pred_ints = self.forecaster.predict_interval(X=X, fh=fh, coverage=coverage)

        return y_pred, y_pred_ints

    def update(self, 
               newdata: pd.DataFrame, 
               fh: Optional[ForecastingHorizon] = None, 
               coverage: float = 0.9, 
               refit: bool = False
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update cutoff value to forecast new dates.
        Possibility to refit the model.

        Parameters
        
            newdata : pd.DataFrame
                The new data containing the same columns as the original data.
            fh : ForecastingHorizon, optional
                Forecast horizon. Default is None and takes the horizon defined at instantiation.
            coverage : float, optional
                The coverage of the confidence interval. Default is 0.9.
            refit : bool, optional
                If True, the model will be refitted on the new training data. Default is False.

        Returns
        
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


    def __init_test(self,
                    data: pd.DataFrame, 
                    depvar_str: str, 
                    exog_l: Optional[list], 
                    freq: str
                   ) -> None:
        """
        Perform checks for the provided data, dependent variable, exogenous variables, and frequency.

        Parameters
        
            data : pd.DataFrame
                A DataFrame containing the input data for forecasting.
            depvar_str : str
                The column name representing the dependent variable for forecasting.
            exog_l : list or None
                List of exogenous variables for forecasting.
            freq : str
                The frequency of the time series data.

        Returns
        
            None

        Raises
        
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

        Returns the X and y for the rest of forecasting operations.
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

    Parameters
    
        LF : Forecast
            An instance of the Forecast class.

    Returns:
        None
    """

    def __init__(self, LF: Forecast):
        self.__dict__.update(LF.__dict__)


    def plot_train_test(self,
                        labels: List[str] = ["y_train", "y_test"],
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        title: str = 'Train-Test sets',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[float, float] = (15, 6)) -> None:
        """
        Plot the dependent variable separating the train from the test windows.

        Parameters
        
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

        Returns
        
            None
        """
        return plot_series(self.y_train, self.y_test, 
                           labels =labels, 
                           ax = ax, 
                           xlabel = xlabel,
                           ylabel = ylabel,
                           title = title,
                           figsize = figsize)


    def plot_cv_procedure(self,
                          ax: Optional[plt.Axes] = None,
                          labels: List[str] = ["Window", "Forecasting horizon"],
                          ylabel: str = "Window number",
                          xlabel: str = "Time",
                          title: str = "Cross Validation Procedure") -> None:
        """
        Plot the cross-validation procedure.

        Parameters
        
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

        Returns
        
            None
        """

        _train_windows, _test_windows = self._get_windows()
        self._plot_windows(_train_windows, _test_windows,
                           ax = ax,
                           labels = labels,
                           xlabel = xlabel, 
                           ylabel = ylabel,
                           title = title)


    def plot_prediction(self,
                        y_pred: pd.Series,
                        y_pred_ints: Optional[pd.DataFrame] = None,
                        interval_label: str = 'prediction interval',
                        labels: List[str] = ["y_train", "y_pred"],
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        title: str = 'Prediction',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[float, float] = (15, 6)):
        """
        Plot the forecast predictions and the confidence intervals.

        Parameters
        
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

        Returns
        
            None
        """

        y = self.y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-5*len(y_pred):]	
        return plot_series(zoom_y_train, y_pred,
                           labels= labels,
                           pred_interval=y_pred_ints,	
                           interval_label ='prediction interval',
                           xlabel = xlabel,
                           ylabel = ylabel,
                           title = title, 
                           ax = ax,
                           figsize = figsize
                          )

    def plot_prediction_true(self,
                             y_pred: pd.Series,
                             y_pred_ints: Optional[pd.DataFrame] = None,
                             interval_label: str = 'prediction interval',
                             labels: List[str] = ["y_train", "y_true", "y_pred"],
                             xlabel: Optional[str] = None,
                             ylabel: Optional[str] = None,
                             title: str = 'Prediction',
                             ax: Optional[plt.Axes] = None,
                             figsize: Tuple[float, float] = (15, 6)):
        """
        Plot the forecast predictions, true values, and the confidence intervals.

        Parameters
        
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

        Returns
        
            None
        """

        y = self.y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-5*len(y_pred):]
        true_pred_idx = np.intersect1d(y.index, y_pred.index)
        err_msg = 'No overlap between true values and predicted values.\nIf you want to plot prediction alone use the function plot_prediction'
        assert len(true_pred_idx)>0, err_msg
        y_true = self.y[true_pred_idx]

        return plot_series(zoom_y_train, y_true, y_pred,
                           pred_interval=y_pred_ints,
                           interval_label ='prediction interval',
                           labels=labels, 
                           xlabel = xlabel,
                           ylabel = ylabel,
                           title = title, 
                           ax = ax,
                           figsize = figsize)

    def _plot_windows(self,
                      train_windows: List[np.ndarray],
                      test_windows: List[np.ndarray],
                      ax: plt.Axes,
                      labels: List[str],
                      ylabel: str,
                      xlabel: str,
                      title: str) -> None:
        """
        Visualize training and test windows.

        Parameters
        
            train_windows : List[np.ndarray]
                List of training window indices.
            test_windows : List[np.ndarray]
                List of test window indices.
            ax : plt.Axes
                The Axes object for the plot.
            labels : List[str]
                Labels for the plot.
            ylabel : str
                Label for the y-axis.
            xlabel : str
                Label for the x-axis.
            title : str
                The title of the plot.

        Returns
        
            None
        """
        assert len(labels)==2, 'wrong number of labels'
        simplefilter("ignore", category=UserWarning)

        def get_y(length, split):
            # Create a constant vector based on the split for y-axis."""
            return np.ones(length) * split
        reformat_xticks = lambda x: x if x=='' else pd.to_datetime(x).strftime('%Y-%m-%d')

        n_splits = len(train_windows)
        n_timepoints = len(self.y)
        len_test = len(test_windows[0])

        train_color, test_color = sns.color_palette("colorblind")[:2]

        if ax is None:
            f, ax = plt.subplots(figsize=plt.figaspect(0.3))

        for i in range(n_splits):
            train = train_windows[i]
            test = test_windows[i]

            ax.plot(
                np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
            )
            ax.plot(
                train,
                get_y(len(train), i),
                marker="o",
                c=train_color,
                label=labels[0],
            )
            ax.plot(
                test,
                get_y(len_test, i),
                marker="o",
                c=test_color,
                label=labels[1],
            )
        #import pdb; pdb.set_trace()
        xticklabels = [int(item) for item in ax.get_xticks()[1:-1]]
        xticklabels = [''] + [self.y.index[i].strftime('%Y-%m-%d') for i in xticklabels] + ['']
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            xticklabels = xticklabels,
        )
        # remove duplicate labels/handles
        handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
        ax.legend(handles, labels, frameon=False)

        if ax is None:			
            return f, ax
        else:
            return ax


    def _get_windows(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate windows for the cross-validation procedure.

        Returns
        
            tuple:
                A tuple containing two lists: train_windows and test_windows.
        """
        _train_windows = []
        _test_windows = []
        _y_index = self.y.index 
        for i, (_train, _test) in enumerate(self.cv.split(_y_index)):
            _train_windows.append(_train)
            _test_windows.append(_test)
        return _train_windows, _test_windows 


##############################################################################
# Model Fit and Insample Performance
##############################################################################
class ForecastFit:
    """
    Class for fitting the forecast model and computing insample performance metrics.

    Parameters
    
        Forecast : Forecast
            An instance of the Forecast class.

    Returns
    
        None
    """

    def __init__(self, Forecast: Forecast):
        """
        Initialize the ForecastFit object.

        Parameters
        
            Forecast : Forecast
                An instance of the Forecast class.
        """

        self.__dict__.update(Forecast.__dict__)

        self.plot = self.__plot()


    def insample_predictions(self, nsample: int = 100) -> pd.DataFrame:
        """
        Compute the insample predictions for the fitted model.

        Parameters
        
            nsample : int, optional
                The number of samples to compute. Default is 100.

        Returns
        
            pd.DataFrame
                A DataFrame containing the insample predictions.
        """

        assert self.forecaster.is_fitted, 'Fit the forecast on the training window first before you can evaluate the insample performance'

        _cutoffs = np.random.choice(len(self.y_train)-len(self.fh), size=nsample, replace=False)

        cv_in = CutoffSplitter(cutoffs=_cutoffs, window_length=1, fh=self.fh)

        insample_result = []                
        for i, (intrain, intest) in tqdm(enumerate(cv_in.split_series(self.y_train))):
            fh = ForecastingHorizon(intest.index, is_relative=False)

            # Apply forecaster to predict the past
            in_pred = self.forecaster.predict(fh=fh, X=self.X).rename('y_pred').reset_index()
            in_pred['y_true'] = intest.values
            in_pred['Abs_diff'] = (in_pred['y_true'] - in_pred['y_pred']).abs()
            in_pred['horizon'] = np.arange(1, len(in_pred)+1)
            in_pred['cutoff'] = intrain.index[-1]

            insample_result.append(in_pred)

        insample_result_df = pd.concat(insample_result)
        self.insample_result_df = insample_result_df
        return insample_result_df

    def insample_perf(self) -> dict:
        """
        Compute insample performance metrics (RMSE and MAPE) for the fitted model.

        Returns
        
            dict
                A dictionary containing the computed insample performance metrics.
        """

        if 'insample_result_df' in self.__dict__.keys():
            pass
        else:           
            self.insample_result_df = self.insample_predictions()

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

    Parameters
    
        LFF : ForecastFit
            An instance of the ForecastFit class.

    Returns
    
        None
    """

    def __init__(self, LFF: ForecastFit):
        """
        Initialize the ForecastFitPlot object.

        Parameters
        
            LFF : ForecastFit
                An instance of the ForecastFit class.
        """

        self.__dict__.update(LFF.__dict__)
        self.LFF = LFF

    def plot_insample_performance(self,
                                  metric: str = 'RMSE',
                                  title: str = 'Insample Performance'):
        """
        Plot the insample performance metrics.

        Parameters
        
            metric : str, optional
                The performance metric to plot. Default is 'RMSE'.
            title : str, optional
                The title of the plot. Default is 'Insample Performance'.
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


##############################################################################
# Model Out of Sample Evaluation
##############################################################################

class ForecastEval:
    """
    Class for evaluating the forecaster out-of-sample.

    Parameters
    
        Forecast : Forecast
            An instance of the Forecast class.

    Returns
    
        None
    """

    def __init__(self, Forecast: Forecast):
        """
        Initialize the ForecastEval object.

        Parameters
        
            Forecast : Forecast
                An instance of the Forecast class.
        """

        self.__dict__.update(Forecast.__dict__)     

        self.scoring_metrics = [MeanSquaredError(square_root=True),
                                MeanAbsolutePercentageError()]
        _rename_metrics = {
            'test_MeanSquaredError':'RMSE', 
            'test_MeanAbsolutePercentageError':'MAPE'
        }
        print(f"\nStart forecaster {self.forecaster_name} evalution....")
        print(" Depending on the forecaster this step may take couple of minutes. Please don't kill the kernel")
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
        print(f"Evaluation time: {np.around(elapsed_time / 60,3)} minutes")

        convert_horizon = self.oos_eval.apply(self.__eval_horizon, axis=1)
        self.oos_horizon_df = pd.concat(convert_horizon.values)

        self.plot = self.__plot()


    def summary_results(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample forecast results.

        Returns
        
            pd.DataFrame
                A DataFrame containing various summary statistics of the out-of-sample forecasts.
        """

        _summary = {'Number of Folds':  self.oos_eval.shape[0], 
         'Avg Fit time (s)': self.oos_eval.fit_time.mean(),
         'Avg_pred_time (s)': self.oos_eval.pred_time.mean(),
         'Smallest training window': self.oos_eval.len_train_window.min(),
         'Largest training window':self.oos_eval.len_train_window.max(),
         'First cutoff': self.oos_eval.index[0],
         'Last cutoff': self.oos_eval.index[-1],
         'Avg MAPE': self.oos_eval.MAPE.mean(),
         'Avg RMSE': self.oos_eval.RMSE.mean()
        }
        return pd.Series(_summary).to_frame().T

    def summary_horizon(self) -> pd.DataFrame:
        """
        Generate a summary of out-of-sample forecast results per horizon.

        Returns
        
            pd.DataFrame
                A DataFrame containing summary performance metrics (RMSE and MAPE) for each horizon.
        """

        self.oos_horizon_perf = summary_perf(self.oos_horizon_df, grouper='horizon', y_true_col = 'y_test', y_pred_col = 'y_pred')        
        return self.oos_horizon_perf

    def __eval_horizon(self, x):
            _fct = pd.concat([x['y_test'], x['y_pred']], keys=['y_test', 'y_pred'], axis=1)
            _fct['Abs_diff'] = (_fct['y_test'] - _fct['y_pred']).abs()        
            _fct['horizon'] = np.arange(1, len(_fct)+1)    
            _fct['cutoff'] = x.name
            return _fct

    def __plot(self):
        return ForecastEvalPlot(self)

class ForecastEvalPlot:
    """
    Plotting utility class for ForecastEval.

    Parameters
    
        LFE : ForecastEval
            An instance of the ForecastEval class.

    Returns
    
        None
    """

    def __init__(self, LFE: ForecastEval):
        """
        Initialize the ForecastEvalPlot object.

        Parameters
        
            LFE : ForecastEval
                An instance of the ForecastEval class.
        """

        self.__dict__.update(LFE.__dict__)
        self.LFE = LFE

    def plot_oos_score(self,
                       score: str = 'RMSE',
                       xlabel: str = None,
                       ylabel: str = None,
                       title: str = 'Out of Sample Performance - Average on All Horizons',
                       ax: Optional[plt.Axes] = None,
                       figsize: Tuple[float, float] = (15, 6)):
        """
        Plot out-of-sample performance metric historically.

        Parameters
        
            score : str, optional
                The performance metric to plot. Default is 'RMSE'.
            xlabel : str, optional
                Label for the x-axis.
            ylabel : str, optional
                Label for the y-axis.
            title : str, optional
                The title of the plot. Default is 'Out of Sample Performance - Average on All Horizons'.
            ax : plt.Axes, optional
                The Axes object for the plot.
            figsize : Tuple[float, float], optional
                The figure size. Default is (15, 6).
        """

        assert score in self.oos_eval.columns, 'score not computed'
        if ax is None:
            f, ax = plt.subplots(1,1,figsize=figsize)

        ylabel = score if ylabel is None else ylabel
        xlabel = '' if xlabel is None else xlabel
        self.oos_eval[score].plot(ax = ax, style = '-o')
        ax.set(xlabel = xlabel, ylabel=ylabel, title =title)

        if ax is None:
            return f, ax
        else:
            return ax

    def plot_oos_horizon(self,
                         score: str = 'RMSE',
                         xlabel: str = None,
                         ylabel: str = None,
                         title: str = 'Out of Sample Performance - Average per Horizons',
                         ax: Optional[plt.Axes] = None,
                         figsize: Tuple[float, float] = (15, 6)):
        """
        Plot out-of-sample performance metric per horizon.

        Parameters
        
            score : str, optional
                The performance metric to plot. Default is 'RMSE'.
            xlabel : str, optional
                Label for the x-axis.
            ylabel : str, optional
                Label for the y-axis.
            title : str, optional
                The title of the plot. Default is 'Out of Sample Performance - Average per Horizons'.
            ax : plt.Axes, optional
                The Axes object for the plot.
            figsize : Tuple[float, float], optional
                The figure size. Default is (15, 6).
        """
        if 'oos_horizon_perf' not in self.__dict__.keys():
            oos_horizon_perf = self.LFE.summary_horizon()

        assert score in oos_horizon_perf.columns, 'score not computed'

        if ax is None:
            f, ax = plt.subplots(1,1,figsize=figsize)

        ylabel = score if ylabel is None else ylabel
        xlabel = '' if xlabel is None else xlabel
        
        oos_horizon_perf[score].plot(ax = ax, style = '-o')
        ax.set(xlabel = xlabel, ylabel=ylabel, title =title)

        if ax is None:
            return f, ax
        else:
            return ax

