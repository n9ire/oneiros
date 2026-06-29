"""
XGBoost training for Oneiros — classification and regression.
Accepts pre-split X_train/X_val/y_train/y_val arrays plus a config dict.
Returns metrics, feature importance, and serialised model bytes.
"""

from __future__ import annotations

import base64
import io
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


def _evals_history(model: Any) -> list[dict]:
    evals: list[dict] = []
    if not hasattr(model, "evals_result_"):
        return evals
    res = model.evals_result_
    keys = list(res.keys())
    if len(keys) < 2:
        return evals
    train_key, val_key = keys[0], keys[1]
    metric_key = list(res[train_key].keys())[0]
    for i, (tl, vl) in enumerate(
        zip(res[train_key][metric_key], res[val_key][metric_key])
    ):
        evals.append({"round": i + 1, "trainLoss": round(tl, 6), "valLoss": round(vl, 6)})
    return evals


def _save_model(model: Any) -> str:
    buf = io.BytesIO()
    model.save_model(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


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

    model = xgb.XGBClassifier(**kw)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    best_iter = int(getattr(model, "best_iteration", kw["n_estimators"] - 1))

    return {
        "task":              "classification",
        "trainAccuracy":     float(model.score(X_train, y_train)),
        "valAccuracy":       float(model.score(X_val, y_val)),
        "bestIteration":     best_iter + 1,
        "nEstimators":       kw["n_estimators"],
        "nClasses":          n_classes,
        "featureImportance": _feat_importance(model, feature_names),
        "evals":             _evals_history(model),
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

    model = xgb.XGBRegressor(**kw)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

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
        "evals":             _evals_history(model),
        "modelB64":          _save_model(model),
    }
