import pandas as pd
from typing import Union, Optional, Tuple, List, Dict, Any

from warnings import simplefilter
simplefilter('ignore')

from .forecast import Forecast
from .forecast import CommonForecastingModels
from forecast_combine.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon


import matplotlib.pyplot as plt
plt.rc('axes', titlesize='x-large')    
plt.rc('axes', labelsize='large')   
plt.rc('xtick', labelsize='large')   
plt.rc('ytick', labelsize='large')   
plt.rc('legend', fontsize='large')   
plt.rc('figure', titlesize='x-large') 
import seaborn as sns
sns.set_theme(style='white', font_scale=1)

class ForecastModelSelect:
    """
    Class for model selection and comparison based on out-of-sample performance.
    Evaluate how to best combine the different models based on their oos performance.

    The class can be initialized in 2 ways. 
        Method 1: Requires a list of Forecast objects. It is the preferable instantiation if the models have already been evaluated out of sample.

        Method 2: Requires all the arguments taken by Forecast plus a dictionary where the keys are names of forecasts and the values are the forecast_models.

    Parameters:
    -----------    
        models_d : dict, optional
            A dictionary containing various forecasting models for comparison. 
            Default is None and assess the most common forecasting models.
                Naive: NaiveForecaster - Keep the latest value
                AutoARIMA: StatsForecastAutoARIMA - Auto ARIMA model
                AutoETS: StatsForecastAutoETS - Auto ETS model
                AutoTheta: StatsForecastAutoTheta - Auto Theta model
                TBATS: StatsForecastAutoTBATS - TBATS model
                LOESS: StatsForecastMSTL - LOESS model
                Prophet: Prophet - Prophet model
        trained_models_d : dict, optional
            A dictionary containing trained Forecast objects. Default is None.
        mode : str, optional
            The aggregation mode. Default is 'nbest_average_horizon'.
        score : str, optional
            The performance score. Default is 'RMSE'.
        nbest : int, optional
            Number of best models to aggregate. Default is 2.
        kwargs
            Additional keyword arguments to be passed to the Forecast initialization.

    Raises:
    -------    
        AssertionError
            If the mode, score, or nbest values are not recognized or do not meet the requirements.
            If both LF_list and models_d are None.
    """

    def __init__(self, 
                 models_d: Optional[Dict] = None,
                 trained_models_d: Optional[Dict] = None,
                 mode: str = 'best_horizon',
                 score: str = 'RMSE',
                 nbest: int = None,
                 **kwargs: Any
                ) -> None:

        lf_d = {}
        if (trained_models_d is not None):
            # Trained Models must be istance of Forecast object
            assert len(set(models_d.keys()).intersection(set(trained_models_d.keys()))) == 0, 'There is an overlap between the two trained and non trainded dictionaries' 
            if (models_d is not None):
                # There must be no overlap between the two dictionaries
                assert all([isinstance(_lf, Forecast) for _lf in trained_models_d.values()]), 'Trained models must be instances of Forecast object'
            for forecaster_name, _lf in trained_models_d.items():
                lf_d[forecaster_name] = _lf        
        else:
            if (models_d is None):
                models_d = CommonForecastingModels

        if models_d is not None:
            for forecaster_name, forecaster in models_d.items():            
                _lf =  Forecast(
                    forecaster_name = forecaster_name,
                    forecaster = forecaster,
                    **kwargs
                )
                lf_d[forecaster_name] = _lf
                           
        self.LF_d = lf_d

        assert isinstance(mode, str), 'mode must be a sting'
        recog_modes = ['best', 'best_horizon', 'average', 'inverse_score', 
                           'nbest_average', 'nbest_average_horizon']
        assert mode in recog_modes, f'mode function not implemented!. Recognized modes are {recog_modes}'
        self.mode = mode

        recog_scores = ['RMSE', 'MAE', 'MAPE', 'R2', 'MedianAE']
        assert score in recog_scores, f'performance score not implemented!. Recognized scores are {recog_scores}'		
        self.score = score

        if nbest is None:
            nbest = 2
        else:
            assert isinstance(nbest, int), 'n best must be an integer'		
        self.nbest = min(nbest, len(self.LF_d))	

        self.eval_models = None
        self.summary_horizon = None
        self.summary_results = None
        self.best_x_overall = None
        self.model_rank_perhorizon = None
        self.avg_oos_horizon = None
        self.avg_oos_hist = None
        return None

    def split_procedure_summary(self):
        """
        Print the summary of the split procedure for each model.

        Returns:
        --------        
            dict:
                A dictionary containing details of the cross-validation procedure, including:
                the number of folds, initial window size, step length, and forecast period.
        """
        _lf  = list(self.LF_d.values())[0]
        return _lf.split_procedure_summary()
    
    def add_forecaster(self, 
                       forecaster_name: str,
                       lf: Forecast) -> None:
        """
        Add a Forecast Object to the list of models to evaluate.

        Parameters:
        -----------
            lf : Forecast
                The Forecast object to be added.
        """
        assert forecaster_name not in self.LF_d.keys(), 'Forecaster name already exists in the considered list of model'
        self.LF_d[forecaster_name] = lf
        return self.LF_d

    def fit(self, 
            on: str = 'all', 
            fh: Optional[ForecastingHorizon] = None, 
            force: bool = False
           ) -> None:
        """
        Fit the forecasting models for all the underlying Forecast objects.

        Parameters:
        -----------        
            on : str, optional
                The period for the in-sample fitting. Default is 'all'.
            fh : ForecastingHorizon, optional
                The forecasting horizon. Default is None and it takes the value entered at initialization..
            force : bool, optional
                If True, force fitting even if the models are already fitted. Default is False.

        Returns:
        --------        
            None
        """
        for _lf in self.LF_d.values():
            if (_lf.is_fitted) and force ==False:
                pass
            else:
                _lf.fit(on=on, fh=fh)
        return None

    def evaluate(self,
                 force: bool = False
                ) -> Tuple:
        """
        Evaluate the underlying Forecast models out of sample.

        Parameters:
        -----------
            force : bool, optional
                If True, force the evaluation even if the models are already evaluated. Default is False.

        Returns:
        --------
            tuple
                A tuple containing the summary of out-of-sample performance per horizon and per model.
        """
        #Step 1: Loop on all models and evaluate them
        _model_evals = []
        for _lf in self.LF_d.values():
            if (_lf.is_evaluated) and force ==False:
                _model_evals.append(_lf.eval)
            else:
                _model_evals.append(_lf.evaluate())
        self.eval_models = _model_evals

        # Step 2: Get the OOS Summary of Performance 
        #per horizon and per model
        _sum_h = {}; _sum_T = {}
        for _lf_eval in self.eval_models:            
            _sum_T[_lf_eval.forecaster_name] = _lf_eval.summary_results().squeeze()
            _sum_h[_lf_eval.forecaster_name] = _lf_eval.summary_horizon()

        self.summary_horizon = pd.concat(_sum_h.values(), axis=1, keys = _sum_h.keys())
        self.summary_results = pd.concat(_sum_T.values(), axis=1, keys = _sum_T.keys())
        return self.summary_horizon, self.summary_results

    def select_best(self,
                    score: Optional[str] = None,
                    reestimate = False,
                   ) -> pd.DataFrame:
        """
        Select the best model based on horizon and overall performance.

        Parameters:
        -----------        
            score : str, optional
                The performance metric to use for model comparison. Default is None and it takes the value entered at initialization.
            nbest : int, optional
                The number of best models to select based on horizon performance. Default is None and it takes the value entered at initialization.

        Returns:
        --------        
            pd.DataFrame
                model_rank_perhorizon. Rank of models per horizon based on performance.
        """
        if score is None:
            score = self.score

        if (self.model_rank_perhorizon is None) or (self.avg_oos_horizon is None) or reestimate:

            # if models are not evaluated yet, run evaluate
            cond1 = self.eval_models is None
            cond2 = self.summary_horizon is None
            if cond1 or cond2:	
                print('\nRun evaluate ...')
                self.evaluate()

            nevals = len(self.LF_d)
            nbest = self.nbest
            if nbest > nevals:
                print(f'\nnbest ={nbest} is higher than the number of models evaluated {nevals}')
                print('All models will be considered. The average best models will be equal to the simple average')
                nbest = max(nbest, nevals)

            best_x_overall = {}
            model_rank_perhorizon = {}
            avg_oos_horizon = {}
            _perf_metrics = list(self.summary_horizon.columns.levels[1])
            for _s in _perf_metrics:
                r = self.summary_horizon.unstack().unstack(1)[_s].unstack()

                # Select best x models based on their average performance on all horizons 
                overall_score = r.mean(axis=1).sort_values(ascending=True)
                best_overall = overall_score.index[0]
                _s_best_x_overall = list(overall_score.index[:nbest])

                # Summary Model average
                _avg_oos_horizon = pd.concat([r.loc[best_overall],
                                            r.min(),
                                            r.loc[_s_best_x_overall].mean(),
                                            r.apply(lambda x: x.nsmallest(nbest).mean(), axis=0),
                                            r.mean(), 
                                            ], 
                                            keys = [
                                            'Best Model (over all)',
                                            'Best Model (per horizon)',
                                            f'Best {nbest} Models (over all)',
                                            f'Best {nbest} Models (per horizon)',
                                            'Model Avg (all models)'
                                            ], 
                                            axis = 1)

                avg_oos_horizon[_s] = pd.concat([r.T, _avg_oos_horizon], axis=1)

                # Compute the rank of the models per horizon
                _rank_perhorizon = r.apply(lambda x: x.nsmallest(nevals).index.tolist(), axis=0)\
                    .reset_index(drop=True).rename(index={i: f'Best_{i+1}' for i in range(nevals)})
                model_rank_perhorizon[_s] = _rank_perhorizon
                best_x_overall[_s] = _s_best_x_overall
            
            self.best_x_overall = best_x_overall
            self.model_rank_perhorizon = model_rank_perhorizon
            self.avg_oos_horizon = avg_oos_horizon

        return self.model_rank_perhorizon[score], self.avg_oos_horizon[score]

    def oos_per_cutoff(self,
                       score: Optional[str] = None
                      ) -> pd.DataFrame:
        """
        Calculate and return the out-of-sample performance based on cutoffs.

        Parameters:
        -----------        
            score : str, optional
                The performance metric to use for model comparison. Should be either 'RMSE' or 'MAPE'. Default is None and it takes the value entered at initialization.
            nbest : int, optional
                The number of best models to select based on horizon performance. Default is None and it takes the value entered at initialization.

        Returns:
        --------        
            pd.DataFrame
                A DataFrame containing the out-of-sample performance per horizon and per model based on the specified score and nbest values.
        """    
        if score is None:
            score = self.score
        
        if self.avg_oos_hist is None:
            nbest = self.nbest
            nevals = len(self.LF_d)
            if nbest > nevals:
                print(f'\nnbest ={nbest} is higher than the number of models evaluated {nevals}')
                print(f'The average performance of best models will be computed on {nevals} models instead')
                nbest = max(nbest, nevals)

            if self.eval_models is None:
                print('Run evaluate ...')
                self.evaluate()

            avg_oos_hist = {}
            _perf_metrics = list(self.summary_horizon.columns.levels[1])
            for _s in _perf_metrics:
                _oos = {_lf_eval.forecaster_name: _lf_eval.oos_cutoff_perf[_s] for _lf_eval in self.eval_models}
                _oos = pd.concat(_oos.values(),  axis=1, keys= _oos.keys())

                best_mod = _oos.mean().idxmin()
                nbest_mods = list(_oos.mean().nsmallest(nbest).index)
                _aggs = pd.concat([_oos[best_mod], _oos[nbest_mods].mean(axis=1), _oos.mean(axis=1)], axis = 1,
                                  keys = ['Best Model (over all)', f'Best {nbest} Models (over all)','Model Avg (all models)'])
                avg_oos_hist[_s] = pd.concat([_oos, _aggs], axis=1)
            self.avg_oos_hist = avg_oos_hist
        
        return avg_oos_hist[score]

    def predict(self,
                X: Optional[pd.DataFrame] = None, 
                fh: Optional[ForecastingHorizon] = None,
                coverage: float = 0.9,
                mode: Optional[str] = None,
                score: Optional[str] = None,
                nbest: Optional[int] = None,
                ret_underlying: bool = False
               ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Make forecasts using the specified aggregation mode.

        Parameters:
        -----------        
            X : pd.DataFrame, optional
                The exogenous variables used for prediction. Default is None and it takes the value entered at initialization..
            fh : ForecastingHorizon, optional
                The forecasting horizon. Default is None and it takes the value entered at initialization..
            coverage : float, optional
                The prediction interval coverage. Default is 0.9.
            mode : str, optional
                The aggregation mode for predictions. Default is None and it takes the value entered at initialization.. 
                Available values:
                * 'best': The prediction is based on the best model.
                * 'best_horizon': The prediction is based on the best model for each horizon.
                * 'average': The average of the prediction of all models.
                * 'inverse_score': The weighted average prediction, where weights are inversely proportional to the model performance score.
                * 'nbest_average': Average of the n best models. The n is given by the parameter nbest.
                * 'nbest_average_horizon': Average of the n best models for each horizon. The n is given by the parameter nbest.
            score : str, optional
                The performance metric to use for model comparison. Should be either 'RMSE' or 'MAPE'. Default is None and it takes the value entered at initialization..
            nbest : int, optional
                The number of best models to select based on horizon performance. Default is None and it takes the value entered at initialization..
            ret_underlying : bool, optional
                If True, return the underlying predictions and prediction intervals for each model. Default is False.

        Returns:
        --------        
            tuple
                A tuple containing the aggregated prediction and prediction intervals.
        """
        preds = {}; pred_ints = {}
        for _fname, _lf in self.LF_d.items():
            _y_pred, _y_pred_ints = _lf.predict(X=X, fh=fh, coverage=coverage)
            preds[_fname] = _y_pred
            pred_ints[_fname] =_y_pred_ints

        preds = pd.concat(preds.values(), keys = preds.keys(), axis=1)
        pred_ints = pd.concat(pred_ints.values(), keys = pred_ints.keys(), axis=1)

        return self.__aggregate_pred(mode=mode, preds=preds, pred_ints=pred_ints, score=score, nbest=nbest, ret_underlying=ret_underlying)

    def update(self,
               newdata: pd.DataFrame,
               refit: bool = False, 
               reevaluate:bool = False, 
               fh: Optional[ForecastingHorizon] = None, 
               coverage: float = 0.9, 
               mode: Optional[str] = None,
               score: Optional[str] = None,
               nbest: Optional[int] = None,
               ret_underlying: bool = False
              ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Function to update the prediction for all the models and aggregate them based on the specific mode.

        Parameters:
        -----------        
            newdata : pd.DataFrame
                The new data for updating the predictions.
            refit : bool, optional
                If False, generate the prediction without fitting the model. If True, fits the underlying models again on the entire sample
                Default is False
            reevaluate: bool, optional    
                Reapply the cross-validateion, evaluate the models out of sample, select the best models, then aggregate the predictions according the setup.
                Default is False.
            fh : ForecastingHorizon, optional
                The forecasting horizon. Default is None and it takes the value entered at initialization..
            coverage : float, optional
                The prediction interval coverage. Default is 0.9.
            mode : str, optional
                The aggregation mode for predictions. Default is None and it takes the value entered at initialization..
                Available values:
                * 'best': The prediction is based on the best model.
                * 'best_horizon': The prediction is based on the best model for each horizon.
                * 'average': The average of the prediction of all models.
                * 'inverse_score': The weighted average prediction, where weights are inversely proportional to the model performance score.
                * 'nbest_average': Average of the n best models. The n is given by the parameter nbest.
                * 'nbest_average_horizon': Average of the n best models for each horizon. The n is given by the parameter nbest.
            score : str, optional
                The performance metric to use for model comparison. Should be either 'RMSE' or 'MAPE'. Default is None and it takes the value entered at initialization..
            nbest : int, optional
                The number of best models to select based on horizon performance. Default is None and it takes the value entered at initialization. and it takes the value entered at initialization..
            ret_underlying : bool, optional
                If True, return the underlying predictions and prediction intervals for each model. Default is False.

        Returns:
        -------
            tuple
                A tuple containing the aggregated prediction and prediction intervals.
        """
        preds = {}; pred_ints = {}
        for _lf in self.LF_d.values():
            _y_pred, _y_pred_ints = _lf.update(newdata = newdata, fh=fh, coverage=coverage, refit=refit)
            preds[_lf.forecaster_name] = _y_pred
            pred_ints[_lf.forecaster_name] =_y_pred_ints

        preds = pd.concat(preds.values(), keys = preds.keys(), axis=1)
        pred_ints = pd.concat(pred_ints.values(), keys = pred_ints.keys(), axis=1)

        if reevaluate == False:
            return self.__aggregate_pred(mode=mode, preds=preds, pred_ints=pred_ints, score=score, nbest=nbest, ret_underlying=ret_underlying)
        else:
            self.evaluate(force=True)
            self.select_best(score = score, reestimate=True)
            return self.__aggregate_pred(mode=mode, preds=preds, pred_ints=pred_ints, score=score, nbest=nbest, ret_underlying=ret_underlying)

    def plot_model_compare(self, 
                           score: str = 'RMSE', 
                           view: str = 'horizon', 
                           model_subset: Optional[List[str]] = None,
                           xlabel: Optional[str] = None, 
                           ylabel: Optional[str] = None,
                           title: str = 'Out of Sample Performance', 
                           ax: Optional[plt.Axes] = None, 
                           figsize: Tuple[int, int] = (15, 6)
                          ) -> Union[plt.Figure, plt.Axes]:
        """
        Plot a comparison of models based on their out-of-sample performance.

        Parameters:
        -----------        
            score : str, optional
                The performance metric for comparison. Should be either 'RMSE' or 'MAPE'. Default is 'RMSE'.
            view : str, optional
                The view mode for comparison, either 'horizon' or 'cutoff'. Default is 'horizon'.
            xlabel : str, optional
                The label for the x-axis. Default is an empty string.
            ylabel : str, optional
                The label for the y-axis. Default is an empty string.
            title : str, optional
                The title of the plot. Default is 'Out of Sample Performance'.
            ax : matplotlib.axes._subplots.AxesSubplot, optional
                The matplotlib axes to use for plotting. Default is None.
            figsize : tuple, optional
                The size of the figure in inches (width, height). Default is (15, 6).

        Returns:
        --------        
            matplotlib.figure.Figure or matplotlib.axes._subplots.AxesSubplot
                The figure and axes of the plot.
        """
        
        if view =='horizon':
            if self.avg_oos_horizon is None:
                self.select_best()
            toplot = self.avg_oos_horizon[score]

        elif view=='cutoff':
            if self.avg_oos_hist is None:
                self.oos_per_cutoff()
            toplot = self.avg_oos_hist[score]

        else:
            raise ValueError('view can take only 2 values: horizon or cutoff')

        if model_subset is not None:
            assert all([m in toplot.columns for m in model_subset]), 'Some models are not in the list of models'
            toplot = toplot[model_subset]
        if ax is None:
            f, ax = plt.subplots(1,1,figsize=figsize)
        if ylabel is None:
            ylabel = score
        if xlabel is None:
            xlabel = view

        toplot.plot(ax=ax, style = '-o')
        ax.set(xlabel = xlabel, ylabel=ylabel)
        ax.set_title(title, size="xx-large")
        ax.legend(frameon=False, bbox_to_anchor=(1, 1))
        if ax is None:
            return f, ax
        else:
            return ax

    def plot_prediction(self, 
                        y_pred: pd.Series, 
                        models_preds: Optional[pd.DataFrame] = None, 
                        y_pred_interval: Optional[Tuple] = None, 
                        interval_label: str = 'CI', 
                        aggregation_label: str = 'Model Agg', 
                        xlabel: str = '', 
                        ylabel: str = '', 
                        title: str = 'Prediction', 
                        ax: Optional[plt.Axes] = None, 
                        figsize: Tuple[int, int] = (15, 6)

                       ) -> Union[plt.Figure, plt.Axes]:
        """
        Plot forecast predictions and aggregated prediction.

        Parameters:
        -----------        
            y_pred : pd.Series
                The aggregated prediction.
            models_preds : pd.DataFrame, optional
                The DataFrame containing predictions from different models. Default is None.
            y_pred_interval : tuple, optional
                Prediction interval for the aggregated prediction. Default is None.
            interval_label : str, optional
                Label for the prediction interval. Default is 'prediction interval'.
            aggregation_label : str, optional
                Label for the aggregated prediction. Default is 'Model Agg'.
            xlabel : str, optional
                Label for the x-axis. Default is an empty string.
            ylabel : str, optional
                Label for the y-axis. Default is an empty string.
            title : str, optional
                Title of the plot. Default is 'Prediction'.
            ax : matplotlib.axes._subplots.AxesSubplot, optional
                Matplotlib axes to use for plotting. Default is None.
            figsize : tuple, optional
                Size of the figure in inches (width, height). Default is (15, 6).

        Returns:
        --------        
            matplotlib.figure.Figure or matplotlib.axes._subplots.AxesSubplot
                The figure and axes of the plot.
        """

        y = list(self.LF_d.values())[0].y
        y_train = y.loc[y.index<y_pred.index[0]]
        zoom_y_train = y_train.iloc[-5*len(y_pred):]
    
        if models_preds is not None:
            model_names = models_preds.columns
            models_preds = [models_preds[c] for c in model_names]
            labels = ['y'] + list(model_names) + [aggregation_label]

            plt_res = plot_series(zoom_y_train, *models_preds, y_pred,
                               labels =labels,
                               xlabel =xlabel,
                               ylabel =ylabel,
                               title =title,
                               ax=ax, 
                               figsize =figsize,
                               pred_interval = y_pred_interval,
                               interval_label =interval_label)
        else:
            plt_res = plot_series(zoom_y_train, y_pred,
                               labels =['y', aggregation_label],
                               xlabel =xlabel,
                               ylabel =ylabel,
                               title =title,
                               ax=ax, 
                               figsize =figsize,
                               pred_interval = y_pred_interval,
                              interval_label =interval_label)
        if ax is None:
            f, ax = plt_res
            ax.legend(frameon=False, bbox_to_anchor=(1, 1))
            return f, ax
        else:
            ax = plt_res
            ax.legend(frameon=False, bbox_to_anchor=(1, 1))
            return ax

    def plot_train_test(self, **kwargs: Any):
        """
        Plot the training and test windows for each model.

        Parameters:
        -----------        
            kwargs
                Additional keyword arguments to be passed to the plot function.

        Returns:
        --------        
            fig : plt.Figure
                If ax was None, a new figure is created and returned
                If ax was not None, the same ax is returned with plot added
            ax : plt.Axis             
                Axes containing the plot             
        """
        _lf  = list(self.LF_d.values())[0]
        return _lf.plot.plot_train_test(**kwargs)
    
    def plot_cv_procedure(self,**kwargs: Any):
        """
        Plot the cross-validation procedure for each model.

        Parameters:
        -----------        
            kwargs
                Additional keyword arguments to be passed to the plot function.

        Returns:
        --------
            fig : plt.Figure
                If ax was None, a new figure is created and returned
                If ax was not None, the same ax is returned with plot added
            ax : plt.Axis             
                Axes containing the plot             
        """
        _lf  = list(self.LF_d.values())[0]
        return _lf.plot.plot_cv_procedure(**kwargs)
    
    def __aggregate_pred(self, mode, preds, pred_ints, score=None, nbest=None, ret_underlying=False):

        if score is None:
            score = self.score
        if nbest is None:
            nbest = self.nbest
        if mode is None:
            mode = self.mode

        if  (mode !='average') & (self.best_x_overall is None):
            self.select_best(score = score)

        if mode =='best':
            # returns the prediction of the best model
            y_pred = preds[self.best_x_overall[score][0]]
            y_pred_int = pred_ints[self.best_x_overall[score][0]]

        elif mode =='best_horizon':
            # returns the prediction of the best model for each forecast horizon
            best_horizon = self.model_rank_perhorizon[score].loc['Best_1'].to_dict()
            y_pred = pd.Series([preds[v].iloc[k-1] for k, v in best_horizon.items()], 
                   index=preds.index)
            y_pred_int = pd.concat([pred_ints[v].iloc[k-1].to_frame().T for k, v in best_horizon.items()])

        elif mode == 'average':
            # returns the average prediction of all models
            y_pred = preds.mean(axis=1)
            y_pred_int = pred_ints.unstack().unstack(0).mean(axis=1).unstack().T

        elif mode == 'inverse_score':
            # returns the Average models proportionals to performance score
            _score = self.summary_results.loc[f'Avg {score}'].sort_values()
            _score = _score/_score.sum()

            _weigh_preds =  preds.mul(_score, axis=1).sum(axis=1)
            y_pred = _weigh_preds.astype('float')

            _weigh_preds_ints = 0
            for m, w in _score.to_dict().items():
                _weigh_preds_ints = _weigh_preds_ints+ pred_ints[m]*w
            y_pred_int = _weigh_preds_ints.astype('float')

        elif mode == 'nbest_average':
            # return the average prediction of nbest models
            y_pred = preds[self.best_x_overall[score]].mean(axis=1)
            y_pred_int = pred_ints[self.best_x_overall[score]]\
            .unstack().unstack(0).mean(axis=1).unstack().T

        elif mode == 'nbest_average_horizon':
            # return the average prediction on the nbest models per horizon
            best_horizon = self.model_rank_perhorizon[score].iloc[:nbest].T
            best_horizon = best_horizon.apply(lambda x: x.values, axis=1).to_dict()

            y_pred = pd.Series([preds[v].iloc[k-1].mean() for k, v in best_horizon.items()], 
                               index=preds.index)

            y_pred_int = pd.concat([pred_ints[v].iloc[k-1].unstack(0).mean(axis=1).to_frame().T for k, v in best_horizon.items()])
            y_pred_int.index = y_pred.index

        else:
            recog_modes = ['best', 'best_horizon', 'average', 'inverse_score', 
                           'nbest_average', 'nbest_average_horizon']
            _error_msg = f'Aggregation mode not recognized. Recognized prediction aggregation are {recog_modes}'
            raise ValueError(_error_msg)

        if ret_underlying:
            return y_pred, y_pred_int, preds, pred_ints
        else:
            return y_pred, y_pred_int

