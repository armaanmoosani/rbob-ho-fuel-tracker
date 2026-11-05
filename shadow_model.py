"""Frozen Huber forecast implementation. Never used to send purchasing alerts."""

import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_huber(x, y, parameters):
    estimator = make_pipeline(StandardScaler(), HuberRegressor(**parameters))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
    return estimator


def forecast(pairs, move, settings):
    """Use sealed inputs only, with a purge and prequential error calibration."""
    if sklearn.__version__ != settings["sklearn_version"]:
        raise ValueError("Shadow sklearn version differs from the frozen trial")
    window = settings["window"]
    calibration = settings["calibration_window"]
    purge = settings["purge"]
    needed = window + calibration + 2 * purge
    if len(pairs) != needed:
        raise ValueError(f"Expected exactly {needed} sealed historical pairs")
    dates = [row[0] for row in pairs]
    if dates != sorted(set(dates)):
        raise ValueError("Shadow training dates must be unique and chronological")
    values = np.asarray([row[1:] for row in pairs], dtype=float)
    if values.shape != (needed, 2) or not np.isfinite(values).all() or not np.isfinite(move):
        raise ValueError("Invalid shadow training values")
    x, y = values.T
    end = len(pairs) - purge
    parameters = settings["huber_parameters"]
    fitted = fit_huber(x[end - window:end], y[end - window:end], parameters)
    point = float(fitted.predict([[move]])[0])
    errors = []
    calibration_dates = []
    for index in range(end - calibration, end):
        train_end = index - purge
        prior = fit_huber(x[train_end - window:train_end], y[train_end - window:train_end], parameters)
        errors.append(float(y[index] - prior.predict([[x[index]]])[0]))
        calibration_dates.append(dates[index])
    errors = np.asarray(errors)
    p_up = float((np.count_nonzero(errors > -point) + 1) / (calibration + 2))
    p_down = float((np.count_nonzero(errors < -point) + 1) / (calibration + 2))
    low, high = np.quantile(errors, [0.1, 0.9]) + point
    probability = p_up if point > 0 else p_down
    direction = 0
    band = "FLAT"
    if abs(move) >= settings["noise_floor_cents"] and point != 0 and probability >= settings["lean_confidence"]:
        direction = 1 if point > 0 else -1
        band = "FULL" if probability >= settings["full_confidence"] else "LEAN"
    scale, huber = fitted.steps[0][1], fitted.steps[1][1]
    slope = float(huber.coef_[0] / scale.scale_[0])
    intercept = float(huber.intercept_ - slope * scale.mean_[0])
    return {
        "point_prediction_cents": point, "p_up": p_up, "p_down": p_down,
        "direction": direction, "band": band,
        "experimental_interval_80": [float(low), float(high)],
        "interval_is_operationally_validated": False,
        "fit_start": dates[end - window], "fit_end": dates[end - 1],
        "calibration_start": calibration_dates[0], "calibration_end": calibration_dates[-1],
        "calibration_errors_cents": errors.tolist(),
        "slope": slope, "intercept_cents": intercept,
        "sklearn_version": sklearn.__version__,
    }
