"""
XGBoost training for Oneiros — classification and regression.
Accepts pre-split X_train/X_val/y_train/y_val arrays plus a config dict.
Returns metrics, feature importance, and serialised model bytes.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Any

import numpy as np

try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def _check_xgb() -> None:
    if not XGB_AVAILABLE:
        raise RuntimeError(
            "xgboost / scikit-learn not installed. Run: pip install xgboost scikit-learn"
        )


def _common_kwargs(config: dict) -> dict[str, Any]:
    early_stop = int(config.get("earlyStoppingRounds", 20))
    kw: dict[str, Any] = dict(
        n_estimators     = int(config.get("nEstimators", 200)),
        max_depth        = int(config.get("maxDepth", 6)),
        learning_rate    = float(config.get("learningRate", 0.1)),
        subsample        = float(config.get("subsample", 0.8)),
        colsample_bytree = float(config.get("colsampleBytree", 0.8)),
        min_child_weight = int(config.get("minChildWeight", 1)),
        gamma            = float(config.get("gamma", 0.0)),
        reg_alpha        = float(config.get("regAlpha", 0.0)),
        reg_lambda       = float(config.get("regLambda", 1.0)),
        random_state     = 42,
        verbosity        = 0,
        n_jobs           = 1,
    )
    if early_stop > 0:
        kw["early_stopping_rounds"] = early_stop
    return kw


def _feat_importance(model: Any, feature_names: list[str]) -> list[dict]:
    importances = model.feature_importances_.tolist()
    return sorted(
        [{"name": n, "importance": round(float(v), 6)} for n, v in zip(feature_names, importances)],
        key=lambda d: d["importance"],
        reverse=True,
    )[:30]


def _evals_history(model: Any, task: str = "classification") -> list[dict]:
    """Extract per-round loss and (for classification) accuracy from evals_result_."""
    evals: list[dict] = []
    if not hasattr(model, "evals_result_"):
        return evals
    res = model.evals_result_
    keys = list(res.keys())
    if len(keys) < 2:
        return evals
    train_key, val_key = keys[0], keys[1]
    train_metrics = res[train_key]
    val_metrics = res[val_key]

    loss_keys = ["logloss", "mlogloss", "rmse", "mae", "rmsle", "tweedie-nloglik@1.5"]
    loss_key = next((k for k in loss_keys if k in train_metrics), list(train_metrics.keys())[0])

    acc_keys = ["error", "merror"]
    acc_key = next((k for k in acc_keys if k in train_metrics), None)

    n_rounds = len(train_metrics[loss_key])
    for i in range(n_rounds):
        entry: dict[str, Any] = {
            "round": i + 1,
            "trainLoss": round(float(train_metrics[loss_key][i]), 6),
            "valLoss": round(float(val_metrics[loss_key][i]), 6),
        }
        if task == "classification" and acc_key:
            entry["trainAccuracy"] = round(1.0 - float(train_metrics[acc_key][i]), 6)
            entry["valAccuracy"] = round(1.0 - float(val_metrics[acc_key][i]), 6)
        evals.append(entry)
    return evals


def _save_model(model: Any) -> str:
    # XGBoost 3.x requires a file path (BytesIO is not supported).
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        model.save_model(path)
        with open(path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("ascii")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _validate_data(data: dict, task: str) -> None:
    """Validate training payload before fit; raises ValueError with actionable messages."""
    for key in ("X_train", "y_train", "X_val", "y_val"):
        if key not in data or data[key] is None:
            raise ValueError(f"Missing '{key}' in training payload.")

    try:
        X_train = np.array(data["X_train"], dtype=np.float32)
        y_train = np.array(data["y_train"])
        X_val = np.array(data["X_val"], dtype=np.float32)
        y_val = np.array(data["y_val"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training arrays: {exc}") from exc

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be 2D arrays (samples × features).")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]} features, val has {X_val.shape[1]}."
        )
    if X_train.shape[1] == 0:
        raise ValueError("Feature matrix has zero columns. Encode categoricals or add numeric features.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train has {X_train.shape[0]} rows but y_train has {y_train.shape[0]}."
        )
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError(
            f"X_val has {X_val.shape[0]} rows but y_val has {y_val.shape[0]}."
        )
    if X_train.shape[0] < 2:
        raise ValueError("Need at least 2 training samples after the pipeline split.")
    if X_val.shape[0] < 1:
        raise ValueError(
            "Validation set is empty. Lower the train ratio in the Split node or add more rows."
        )
    if np.isnan(X_train).any() or np.isnan(X_val).any():
        raise ValueError(
            "Feature matrix contains NaN values. Add a Fill NaN step in the Dataset pipeline."
        )

    if task == "regression":
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
            raise ValueError("Regression targets must be finite numbers.")
    else:
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        classes = np.unique(np.concatenate([y_train, y_val]))
        if len(classes) < 2:
            raise ValueError("Classification requires at least 2 classes in train and validation labels.")


def _xgb_hint(exc: Exception) -> str:
    msg = str(exc).lower()
    if "nan" in msg:
        return "Add a Fill NaN transform in Dataset → Pipeline."
    if "validation set is empty" in msg or "train ratio" in msg:
        return "Lower the train ratio on the Split node or add more rows."
    if "zero columns" in msg or "no numeric feature" in msg:
        return "One-hot encode categoricals or drop non-numeric columns in the pipeline."
    if "at least 2 classes" in msg:
        return "Pick a target with multiple classes or switch to Regression."
    if "regression targets" in msg or "finite" in msg:
        return "Use a numeric target column and set Task to Regression."
    if "early stopping" in msg or "n_estimators" in msg:
        return "Set early_stopping_rounds lower than n_estimators."
    if "not installed" in msg:
        return "Run: pip install xgboost scikit-learn"
    return "Check your dataset, target column, and pipeline in the Dataset tab."


# ── Public entry point ────────────────────────────────────────────────────────

def train_xgboost(data: dict, config: dict) -> dict[str, Any]:
    """
    Train XGBoost (classification or regression).

    config extra keys:
        task        str  'classification' | 'regression'  (default 'classification')
        objective   str  objective override (e.g. 'reg:squarederror')
    """
    _check_xgb()

    task: str = config.get("task", "classification")
    _validate_data(data, task)

    if task == "regression":
        return _train_regression(data, config)
    return _train_classification(data, config)


# ── Classification ────────────────────────────────────────────────────────────

def _train_classification(data: dict, config: dict) -> dict[str, Any]:
    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"], dtype=np.int32)
    X_val   = np.array(data["X_val"],   dtype=np.float32)
    y_val   = np.array(data["y_val"],   dtype=np.int32)

    feature_names: list[str] = data.get("featureNames") or [f"f{i}" for i in range(X_train.shape[1])]
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))

    obj = config.get("objective") or ("binary:logistic" if n_classes == 2 else "multi:softmax")
    metric = ["logloss", "error"] if n_classes == 2 else ["mlogloss", "merror"]

    kw = _common_kwargs(config)
    kw["objective"]    = obj
    kw["eval_metric"]  = metric
    if n_classes > 2:
        kw["num_class"] = n_classes

    try:
        model = xgb.XGBClassifier(**kw)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    except Exception as exc:
        raise RuntimeError(f"XGBoost classification failed: {exc}") from exc

    best_iter = int(getattr(model, "best_iteration", kw["n_estimators"] - 1))

    return {
        "task":              "classification",
        "trainAccuracy":     float(model.score(X_train, y_train)),
        "valAccuracy":       float(model.score(X_val, y_val)),
        "bestIteration":     best_iter + 1,
        "nEstimators":       kw["n_estimators"],
        "nClasses":          n_classes,
        "featureImportance": _feat_importance(model, feature_names),
        "evals":             _evals_history(model, "classification"),
        "modelB64":          _save_model(model),
    }


# ── Regression ────────────────────────────────────────────────────────────────

_REG_OBJECTIVES: dict[str, tuple[str, list[str]]] = {
    "reg:squarederror":  ("Squared Error",    ["rmse"]),
    "reg:absoluteerror": ("Absolute Error",   ["mae"]),
    "reg:squaredlogerror": ("Squared Log Error", ["rmsle"]),
    "reg:tweedie":       ("Tweedie",          ["tweedie-nloglik@1.5"]),
    "reg:pseudohubererror": ("Pseudo-Huber",  ["rmse"]),
}

REGRESSION_OBJECTIVES = [
    {"value": k, "label": v[0]} for k, v in _REG_OBJECTIVES.items()
]


def _train_regression(data: dict, config: dict) -> dict[str, Any]:
    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"], dtype=np.float32)
    X_val   = np.array(data["X_val"],   dtype=np.float32)
    y_val   = np.array(data["y_val"],   dtype=np.float32)

    feature_names: list[str] = data.get("featureNames") or [f"f{i}" for i in range(X_train.shape[1])]

    obj = config.get("objective") or "reg:squarederror"
    _, metric = _REG_OBJECTIVES.get(obj, ("", ["rmse"]))

    kw = _common_kwargs(config)
    kw["objective"]   = obj
    kw["eval_metric"] = metric

    try:
        model = xgb.XGBRegressor(**kw)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    except Exception as exc:
        raise RuntimeError(f"XGBoost regression failed: {exc}") from exc

    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    val_rmse   = float(np.sqrt(mean_squared_error(y_val,   val_pred)))
    train_mae  = float(mean_absolute_error(y_train, train_pred))
    val_mae    = float(mean_absolute_error(y_val,   val_pred))
    val_r2     = float(r2_score(y_val, val_pred))

    best_iter = int(getattr(model, "best_iteration", kw["n_estimators"] - 1))

    return {
        "task":              "regression",
        "objective":         obj,
        "trainRMSE":         round(train_rmse, 6),
        "valRMSE":           round(val_rmse,   6),
        "trainMAE":          round(train_mae,  6),
        "valMAE":            round(val_mae,    6),
        "valR2":             round(val_r2,     4),
        "bestIteration":     best_iter + 1,
        "nEstimators":       kw["n_estimators"],
        "featureImportance": _feat_importance(model, feature_names),
        "evals":             _evals_history(model, "regression"),
        "modelB64":          _save_model(model),
    }
