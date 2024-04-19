from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('axes', titlesize='large')    
plt.rc('axes', labelsize='large')   
plt.rc('xtick', labelsize='large')   
plt.rc('ytick', labelsize='large')   
plt.rc('legend', fontsize='large')   
plt.rc('figure', titlesize='x-large') 
import seaborn as sns
sns.set_theme(style='white', font_scale=1)


def plot_time_series(series:pd.Series,
                     title:str = 'Time Series Plot',
                     xlabel:str = 'Date',
                     ylabel:str = 'Value'):
    """
    Plot time series data with date on x-axis and value on y-axis.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        title: str
            Title of the plot
        xlabel: str
            Label for x-axis
        ylabel: str
            Label for y-axis

    Returns:
    --------
        None    
    """
    plt.figure(figsize=(10, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_acf_pacf(series:pd.Series,
                 lags:int=40,
                 title:str='ACF and PACF Plots'):
    """
    Plot ACF and PACF plots for the given time series data.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        lags: int
            Number of lags to consider
        title: str
            Title of the plot

    Returns:
    --------
        None    
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(series, ax=ax[0], lags=lags)
    plot_pacf(series, ax=ax[1], lags=lags)
    fig.suptitle(title)
    plt.show()

def test_stationarity(series: pd.Series):
    """
    Perform Dickey-Fuller test for stationarity of the given time series data.

    Parameters:
    -----------
        series: pd.Series
            Time series data

    Returns:
    --------
        None    
    """
    result = adfuller(series.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    # print whether the series is stationary or not at 5%
    print("Is the series stationary? {0}".format('Yes' if result[1] < 0.05 else 'No'))

def decompose_series(series:pd.DataFrame,
                     period:int=12,
                     title:str='Time Series Decomposition'):
    """
    Decompose the time series data into trend, seasonal and residual components.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        period: int
            Period of the seasonal component
        title: str
            Title of the plot

    Returns:
    --------
        decomposition: statsmodels.tsa.seasonal.DecomposeResult
            Decomposed time series data    
    """
    decomposition = seasonal_decompose(series, model='additive', period=period)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(title)
    plt.show()
    return decomposition

# Plot yearly seasonality
def plot_yearly_seasonality(series: pd.DataFrame,
                            title:str='Yearly Seasonality',
                            show_all:bool = True):
    """
    Plot yearly seasonality of the given time series data.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        title: str
            Title of the plot
        show_all: bool
            Whether to show all yearly patterns or just the average

    Returns:
    --------
        None
    """

    df = series.to_frame("value")
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear

    yearly_patterns = df.pivot_table(values='value', index='day_of_year', columns='year')

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Day of Year")
    plt.ylabel("Values")
    if show_all:
        for column in yearly_patterns:
            plt.plot(yearly_patterns.index, yearly_patterns[column], 
                     label=f'Year {column}', marker='o', linestyle='-')
        plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    else:
        plt.plot(yearly_patterns.index, yearly_patterns.mean(axis=1),
                  label='Average', marker='o', linestyle='-')
        plt.legend(frameon=False)
    plt.show()

# Remove yearly seasonality and show monthly patterns
def analyze_monthly_pattern(series:pd.Series,
                            title:str='Monthly Seasonality',
                            show_all:bool = False):
    """
    Analyze monthly seasonality of the given time series data after removing yearly seasonality.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        title: str
            Title of the plot
        show_all: bool
            Whether to show all monthly patterns or just the average

    Returns:
    --------
        None
    """
    df = series.to_frame("value")
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day

    # Adjust for yearly seasonality by subtracting the mean for each day of the year
    daily_means = df.groupby(df.index.dayofyear)['value'].transform('mean')
    df['adjusted'] = df['value'] - daily_means

    monthly_patterns = df.pivot_table(values='adjusted', index='day_of_month', columns='month')

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Day of Month")
    plt.ylabel("Adjusted Values")
    if show_all:
        for column in monthly_patterns:
            plt.plot(monthly_patterns.index, monthly_patterns[column], 
                     label=f'Month {column}', marker='o', linestyle='-')
        plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    else:
        plt.plot(monthly_patterns.index, monthly_patterns.mean(axis=1),
                 label='Average', marker='o', linestyle='-')
        plt.legend(frameon=False)
    plt.show()

# Further adjust for monthly seasonality and analyze weekly patterns
def analyze_weekly_pattern(series:pd.Series,
                           title:str='Weekly Seasonality'):
    """
    Analyze weekly seasonality of the given time series data after removing yearly and monthly seasonality.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        title: str
            Title of the plot

    Returns:
    --------
        None
    """
    df = series.to_frame("value")
    df['day_of_week'] = df.index.dayofweek

    # Adjust for yearly seasonality
    daily_means = df.groupby(df.index.month)['value'].transform('mean')
    df['adjusted'] = df['value'] - daily_means

    # Further adjust for monthly seasonality
    monthly_means = df.groupby(df.index.day)['adjusted'].transform('mean')
    df['monthly_adjusted'] = df['adjusted'] - monthly_means

    weekly_patterns = df.pivot_table(values='monthly_adjusted', index='day_of_week')
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Day of Week")
    plt.ylabel("Adjusted Values")
    plt.plot(weekly_patterns.index, weekly_patterns['monthly_adjusted'], marker='o', linestyle='-')
    plt.xticks(weekly_patterns.index, [day_map[day] for day in weekly_patterns.index])
    plt.grid(True)
    plt.show()

def analyse_series(series:pd.Series,
                   series_name:str ='',
                   period:int=12,
                   acf_lags: int=40):
    """
    Wrapper function to all the above functions to analyze the given time series data.

    Parameters:
    -----------
        series: pd.Series
            Time series data
        series_name: str
            Name of the time series
        period: int
            Period of the seasonal component
        acf_lags: int
            Number of lags to consider for ACF and PACF plots

    Returns:
    --------
        None
    """
    plot_time_series(series, title=f'{series_name}')

    # Stationarity test
    print("Results of Dickey-Fuller Test:")
    test_stationarity(series)
    
    # ACF and PACF plots
    plot_acf_pacf(series, lags=acf_lags, title=f'{series_name} ACF and PACF')
   
    # Decompose series
    decompose_series(series, period=period, title=f'{series_name} Decomposition')

    # Analyze yearly seasonality
    plot_yearly_seasonality(series, title=f'{series_name} Yearly Seasonality')

    # Analyze monthly pattern
    analyze_monthly_pattern(series, title=f'{series_name} Monthly Pattern')

    # Analyze weekly pattern
    analyze_weekly_pattern(series, title=f'{series_name} Weekly Pattern')

