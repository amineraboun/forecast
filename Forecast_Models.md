# Most Common Forecasting Models Available in the Package
## Trend forecasters
### TrendForecaster
Trend-based forecast. regresses the values on their index.
For input time series $(v_i, t_i),\quad i=1,...,T$ fits an sklearn model $v_i = f(t_i) + \epsilon_i$

```python
class TrendForecaster(regressor=None):
'''
Define the regression model type. If not set, it will default to sklearn.linear_model.LinearRegression
'''

from sktime.forecasting.trend import TrendForecaster
forecaster = TrendForecaster()
```
### PolynomialTrendForecaster

```python
class PolynomialTrendForecaster(regressor=None, degree=1, with_intercept=True):
'''
regressor sklearn regressor estimator object, default = None
Define the regression model type. If not set, it will default to sklearn.linear_model.LinearRegression

degree int, default = 1
Degree of the polynomial function

with_intercept bool, default=True
If true, include a feature in which all polynomial powers are zero.
(i.e. a column of ones, acts as an intercept term in a linear model)
'''

from sktime.forecasting.trend import TrendForecaster
forecaster = TrendForecaster()
```

## Auto ARIMA

```python
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
```

## Auto ETS
This is a family of methods that try several definition of the ETS (Error, Trend, Seasonality) models and picks us the best

### Simple exponential smoothing:
This method is suitable for forecasting data with no clear trend or seasonal pattern.

Formulation:
`````{admonition} Component Form
:class: definition 
$$
\begin{align*}
Forecast~equation:\quad & \hat{y}_{T+h|T} = \hat{y}_{t+1|t} = l_t\\
Smoothing~equation:\quad & l_t = \alpha Y_t + (1-\alpha)l_{t-1}\\
\end{align*}
$$
where $l_t$ is the level (or the smoothed value) of the series at time $t$
`````

`````{admonition} Weighted Average From
:class: definition 
$$
\begin{align*}
\hat{y}_{T+h|T} =& \quad \alpha y_T + \alpha(1-\alpha)y_{T-1} + \alpha(1-\alpha)^2y_{T-2}+ ...\\
                =& \quad \sum_{j=0}^{T-1} {\alpha(1-\alpha)^j y_{T-j} +(1-\alpha)^Tl_0}
\end{align*}
$$

where  $0 \leq \alpha \leq 1$ is the smoothing parameter. The one-step-ahead forecast for time  T+1 is a weighted average of all of the observations in the series  
The rate at which the weights decrease is controlled by the parameter $\alpha$.
`````

Simple exponential smoothing has a “flat” forecast function. All forecasts take the same value, equal to the last level component.
```math
\hat{y}_{T+h|T} = \hat{y}_{T+1|T} = l_T
```

### With Trend:
Extention by Holt (1957) to allow the forecasts to move with the trend

```math
$$
\begin{align*}
Forecast~equation:\quad & \hat{y}_{t+h|t} =  l_t + h b_t\\
Level~equation:\quad & l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})\\
Trend~equation:\quad & b_t = \beta(l_t-l_{t-1}) + (1-\beta)b_{t-1}
\end{align*}
$$
where $l_t$ is the level (or the smoothed value) of the series at time $t$
```
The level equation shows that the level is a weighted average of observation  $y_t$ and the one-step-ahead training forecast for time $t$, here given by l_{t-1} + b_{t-1}
The trend equation shows that  
b
t
  is a weighted average of the estimated trend at time  t   based on  and  , the previous estimate of the trend.
### Wind Seasonality:


```python
from sktime.forecasting.statsforecast import StatsForecastAutoETS
```
## MSTL
MSTL stands for **M**ultiple **S**easonal-**T**rend decomposition using **L**OESS model.
 1. decomposes the time series in multiple seasonalities using LOESS. 
 Where LOESS means locally estimated scatterplot smoothing. It is a locally fitted regression model that takes into account non-linearities
 
    The decomposition algorithm performs smoothing in two loops:
    * The inner loop iterates between seasonal and trend smoothing. the seasonal component is calculated first and removed to calculate the trend component 
    * The outer loop minimizes the effect of outliers by subtracting the seasonal and trend components from the time series.
 
 2. forecasts the trend using a custom non-seasonal model (trend_forecaster)
 3. forecasts each seasonality using a SeasonalNaive model.

```python
class StatsForecastMSTL(season_length: int | List[int], trend_forecaster=None):
'''
season_length Union[int, List[int]]
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
