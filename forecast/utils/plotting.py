import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import flatten
from matplotlib.ticker import FuncFormatter, MaxNLocator

from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_interval_df, check_y
from sktime.utils.validation.series import check_consistent_index_type
from sktime.datatypes import convert_to



def plot_series(
    *series,
    labels=None,
    markers=None,
    colors=None,
    title=None,
    xlabel=None,
    ylabel=None,
    ax=None,
	figsize =(15,6),
    pred_interval=None,
	interval_label ='prediction interval'
):
    """Plot one or more time series.

    Parameters
    ----------
    series : pd.Series or iterable of pd.Series
        One or more time series
    labels : list, default = None
        Names of series, will be displayed in figure legend
    markers: list, default = None
        Markers of data points, if None the marker "o" is used by default.
        The length of the list has to match with the number of series.
    colors: list, default = None
        The colors to use for plotting each series. Must contain one color per series
    title: str, default = None
        The text to use as the figure's suptitle
    pred_interval: pd.DataFrame, default = None
        Output of `forecaster.predict_interval()`. Contains columns for lower
        and upper boundaries of confidence interval.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis

    Examples
    --------
    >>> from sktime.utils.plotting import plot_series
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_series(y)  # doctest: +SKIP
    """
    
    _check_soft_dependencies("matplotlib", "seaborn")
    
    for y in series:
        check_y(y)

    series = list(series)
    series = [convert_to(y, "pd.Series", "Series") for y in series]

    n_series = len(series)
    _ax_kwarg_is_none = True if ax is None else False
    # labels
    if labels is not None:
        if n_series != len(labels):
            raise ValueError(
                """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
            )
        legend = True
    else:
        labels = ["" for _ in range(n_series)]
        legend = False

    # markers
    if markers is not None:
        if n_series != len(markers):
            raise ValueError(
                """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
            )
    else:
        markers = ["o" for _ in range(n_series)]

    # create combined index
    index = series[0].index
    for y in series[1:]:
        # check index types
        check_consistent_index_type(index, y.index)
        index = index.union(y.index)

    # generate integer x-values
    xs = [np.argwhere(index.isin(y.index)).ravel() for y in series]

    # create figure if no Axe provided for plotting
    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(1, figsize=figsize)

    # colors
    if colors is None or not _check_colors(colors, n_series):
        colors = sns.color_palette("colorblind", n_colors=n_series)

    # plot series
    for x, y, color, label, marker in zip(xs, series, colors, labels, markers):
        # scatter if little data is available or index is not complete
        # if len(x) <= 3 or not np.array_equal(np.arange(x[0], x[-1] + 1), x):
        #     plot_func = sns.scatterplot
        # else:
        plot_func = sns.lineplot

        plot_func(x=x, y=y, ax=ax, marker=marker, label=label, color=color)

    # combine data points for all series
    xs_flat = list(flatten(xs))

    # set x label of data point to the matching index
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in xs_flat:
            return index[int(tick_val)]
        else:
            return ""

	
    # dynamically set x label ticks and spacing from index labels
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	
	# Change the xtickslabels
    reformat_xticks = lambda x: x if x=='' else pd.to_datetime(x).strftime('%Y-%m-%d')
    xticklabels = [reformat_xticks(item.get_text()) for item in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels)
	
    # Set the figure's title
    if title is not None:
        ax.set_title(title, size="xx-large")

    # Label the x and y axes
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    _ylabel = ylabel if ylabel is not None else series[0].name
    ax.set_ylabel(_ylabel)

    ax.legend(loc='best', frameon=False)
	
    if pred_interval is not None:
        check_interval_df(pred_interval, series[-1].index)
        ax = plot_interval(ax, pred_interval, interval_label)
    
    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax

def plot_interval(ax, interval_df, legend_label ='prediction interval'):
    cov = interval_df.columns.levels[1][0]
    var_name = interval_df.columns.levels[0][0]
    ax.fill_between(
        ax.get_lines()[-1].get_xdata(),
        interval_df[var_name][cov]["lower"].astype("float64"),
        interval_df[var_name][cov]["upper"].astype("float64"),
        alpha=0.2,
        color=ax.get_lines()[-1].get_c(),
        label=f"{int(cov * 100)}% {legend_label}",
    )
    ax.legend()
    return ax

