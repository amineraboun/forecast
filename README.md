# forecast

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

forecast is a Python library built upon the foundation of the sktime library, designed to simplify and streamline the process of forecasting and prediction model aggregation. It provides tools for aggregating predictions from multiple models, evaluating their performance, and visualizing the results. Whether you're working on time series forecasting, data analysis, or any other predictive modeling task, forecast offers a convenient and efficient way to handle aggregation and comparison.

## Key Features

- **Model Aggregation:** Easily aggregate predictions from multiple models using various aggregation modes such as best model overall, best model per horizon, inverse score weighted average model, and more.
- **Out-of-Sample Evaluation:** Evaluate model performance using out-of-sample data and choose the best models based on user-defined performance metrics.
- **Visualization:** Visualize model performance, aggregated predictions, and prediction intervals with built-in plotting functions.
- **Flexibility:** Accommodate various aggregation strategies, forecast horizons, and performance metrics to cater to your specific use case.

## Installation

Install Your Package Name using pip:

```bash
pip install your-package-name
```

## Usage

```python
# Import the necessary classes from your-package-name
from forecast.model_select import YourClassName

# Create an instance of YourClassName
model = YourClassName()

# Make predictions using the predict method
predictions = model.predict(X, fh, mode='best', score='RMSE')

# Update predictions with new data using the update method
updated_predictions = model.update(newdata, refit=True, mode='average')

# Visualize model comparison
model.plot_model_compare(score='RMSE', view='horizon')
model.plot_model_compare(score='RMSE', view='cutoff')

# Visualize model prediction
model.plot_prediction(predictions, models_preds=None, title='Prediction')
```

## Documentation

For detailed information about available classes, methods, and parameters, please refer to the [Documentation](link-to-your-documentation).

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

We welcome contributions from the community! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request. 

## Contact

For queries, support, or general inquiries, please feel free to reach me at [amineraboun@gmail.com](mailto:amineraboun@gmail.com).