# Most Common Forecasting Models Available in the Package
## Trend
Trend based forecast. regresses the values on their index.
For input time series $(v_i, t_i),\quad i=1,...,T$ fits an sklearn model $v_i = f(t_i) + \epsilon_i$

```python
class TrendForecaster(regressor=None):
'''
Define the regression model type. If not set, will default to sklearn.linear_model.LinearRegression
'''

from sktime.forecasting.trend import TrendForecaster
forecaster = TrendForecaster()
```

## Auto ARIMA

```python
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
```

## Auto ETS

## MSTL
MSTL stands for **M**ultiple **S**easonal-**T**rend decomposition using **L**OESS model.
 1. decomposes the time series in multiple seasonalities using LOESS. 
 Where LOESS means locally estimated scatterplot smoothing. It is a locally fitted regression model that takes into account non linearities
 
    The decomposition algorithm performs smoothing in two loops:
    * The inner loop iterates between seasonal and trend smoothing. the seasonal component is calculated first and removed to calculate the trend component 
    * The outer loop minimizes the effect of outliers by subtracting the seasonal and trend components from the time series.
 
 2. forecasts the trend using a custom non-seasonal model (trend_forecaster)
 3. forecasts each seasonality using a SeasonalNaive model.

```python
class StatsForecastMSTL(season_length: int | List[int], trend_forecaster=None):
'''
season_length
Union[int, List[int]]
Number of observations per unit of time. For multiple seasonalities use a list.

trend_forecaster
estimator, optional, default=StatsForecastAutoETS()
Sktime estimator used to make univariate forecasts. Multivariate estimators are not supported.
'''

from sktime.forecasting.statsforecast import StatsForecastMSTL
forecaster = StatsForecastMSTL()
```
## Auto Theta

## Auto CES

## TBATS

## Prophet

## GARCH