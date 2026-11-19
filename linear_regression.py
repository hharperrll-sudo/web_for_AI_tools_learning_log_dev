"""Fit and use a simple one-variable linear regression model."""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class LinearRegressionModel:
    """A fitted line of the form y = slope * x + intercept."""

    slope: float
    intercept: float

    def predict(self, x_values: Sequence[float]) -> list[float]:
        """Predict target values for a sequence of inputs."""
        return [self.slope * x + self.intercept for x in x_values]


def fit_linear_regression(
    x_values: Sequence[float], y_values: Sequence[float]
) -> LinearRegressionModel:
    """Fit y = slope * x + intercept using the least-squares method."""
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) < 2:
        raise ValueError("at least two data points are required")

    try:
        xs = [float(value) for value in x_values]
        ys = [float(value) for value in y_values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("all data points must be numeric") from error

    if not all(isfinite(value) for value in xs + ys):
        raise ValueError("all data points must be finite numbers")

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance == 0:
        raise ValueError("x_values must contain at least two distinct values")

    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    return LinearRegressionModel(slope=slope, intercept=intercept)


if __name__ == "__main__":
    sample_x = [1, 2, 3, 4, 5, 6]
    sample_y = [5.2, 6.8, 9.1, 10.9, 13.2, 14.8]

    model = fit_linear_regression(sample_x, sample_y)
    next_prediction = model.predict([7])[0]

    print(f"Fitted line: y = {model.slope:.3f}x + {model.intercept:.3f}")
    print(f"Prediction for x = 7: {next_prediction:.3f}")
