# forecast_combine

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

forecast_combine is a Python library built upon the foundation of the sktime library, designed to simplify and streamline the process of forecasting and prediction model aggregation. It provides tools for aggregating predictions from multiple models, evaluating their performance, and visualizing the results. Whether you're working on time series forecasting, data analysis, or any other predictive modeling task, forecast_combine offers a convenient and efficient way to handle aggregation and comparison.

## Key Features

- **Model Aggregation:** Easily aggregate predictions from multiple models using various aggregation modes such as best model overall, best model per horizon, inverse score weighted average model, and more.
- **Out-of-Sample Evaluation:** Evaluate model performance using out-of-sample data and choose the best models based on user-defined performance metrics.
- **Visualization:** Visualize model performance, aggregated predictions, and prediction intervals with built-in plotting functions.
- **Flexibility:** Accommodate various aggregation strategies, forecast horizons, and performance metrics to cater to your specific use case.

## Installation

Install Your Package Name using pip:

```bash
pip install forecast_combine
```

## Usage

```python
# Read the data
import pandas as pd
import numpy as np
data = pd.Series(np.cumsum(np.random.normal(0, 1, size=1000)), 
                 index=pd.date_range(end='31/12/2022', periods=1000)
                ).rename('y').to_frame()

# Import the package
from forecast_combine.model_select import ForecastModelSelect

# Import the packages of the models to test
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import (
    StatsForecastAutoARIMA,
    StatsForecastAutoETS, 
    StatsForecastAutoTheta,
    StatsForecastAutoTBATS
)

# Define the forecasting models 
ForecastingModels = {
    "Naive": NaiveForecaster(),
    "AutoARIMA": StatsForecastAutoARIMA(),
    "AutoETS": StatsForecastAutoETS(),
    "AutoTheta": StatsForecastAutoTheta(),
    "AutoTBATS": StatsForecastAutoTBATS(seasonal_periods=1),
}

model = ForecastModelSelect(
            data= data,
            depvar_str = 'y',                 
            exog_l=None,
            fh = 10,
            pct_initial_window=0.75,
            step_length = 5,
            forecasters_d = ForecastingModels,
            freq = 'B',
            mode = 'best_horizon',
            score = 'RMSE', )

# evaluate all the models out-of-sample
summary_horizon, summary_results = model.evaluate()

# compare models
rank, score = model.select_best(score = 'MAPE')

# Visualize model comparison
model.plot_model_compare(score ='MAPE', view = 'cutoff')
model.plot_model_compare(score ='MAPE', view = 'horizon')

# Generate prediction
y_pred, y_pred_ints, preds, pred_ints =  model.predict(score='RMSE', ret_underlying=True)

# Visualize prediction
model.plot_prediction(y_pred = y_pred,
                     models_preds = preds,
                     y_pred_interval = y_pred_ints, 
                     title = 'Prediction')
```

## Documentation

For detailed information about available classes, methods, and parameters, please refer to the [Documentation](https://amineraboun.github.io/forecast/).

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

We welcome contributions from the community! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request. 

## Contact

For queries, support, or general inquiries, please feel free to reach me at [amineraboun@gmail.com](mailto:amineraboun@gmail.com).
