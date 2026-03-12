import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


# =========================================================
# 1. Basic utilities
# =========================================================

def load_meta(processed_dir: Path) -> dict:
    with open(processed_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "grid_remap" in meta:
        meta["grid_remap"] = {int(k): v for k, v in meta["grid_remap"].items()}
    return meta


def get_neighbors(grid_idx: int, meta: dict) -> list:
    """
    Return the 8-neighborhood indices for the given grid_idx.

    If active_grids/grid_remap are available in meta, use them.
    Otherwise, assume grid indices are continuous in a regular grid.
    """

    # Prefer n_rows / n_cols; otherwise try grid_h / grid_w
    if "n_rows" in meta and "n_cols" in meta:
        n_rows = meta["n_rows"]
        n_cols = meta["n_cols"]
    elif "grid_h" in meta and "grid_w" in meta:
        n_rows = meta["grid_h"]
        n_cols = meta["grid_w"]
    elif "grid_shape" in meta and len(meta["grid_shape"]) == 2:
        n_rows, n_cols = meta["grid_shape"]
    else:
        raise KeyError(
            "Cannot find grid shape information in meta.json, "
            "such as n_rows/n_cols or grid_shape."
        )

    n_grids = meta["n_grids"]

    # Case 1: meta contains active_grids + grid_remap
    if "active_grids" in meta and "grid_remap" in meta:
        active_grids = meta["active_grids"]
        grid_remap = meta["grid_remap"]

        grid_id = active_grids[grid_idx]
        row, col = divmod(grid_id, n_cols)

        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                    continue
                nb_id = nr * n_cols + nc
                if nb_id in grid_remap:
                    neighbors.append(grid_remap[nb_id])
        return neighbors

    # Case 2: grid_idx itself is treated as a regular contiguous grid index
    row, col = divmod(grid_idx, n_cols)

    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                continue
            nb_idx = nr * n_cols + nc
            if 0 <= nb_idx < n_grids:
                neighbors.append(nb_idx)

    return neighbors


def build_neighbor_list(meta: dict):
    return [get_neighbors(g, meta) for g in range(meta["n_grids"])]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


# =========================================================
# 2. Feature configuration
# =========================================================

def get_feature_config():
    """
    There are 6 time slots per day, so:
    6   = same time slot on the previous day
    42  = same time slot one week earlier
    180 = past 30-day window
    """
    config = {
        "lags": [1, 2, 3, 6, 12, 18, 42, 84, 168, 180],
        "rolling_windows": [6, 42, 180],   # rolling mean
        "rolling_std_windows": [42, 180],  # rolling std
        "neighbor_windows": [1, 6, 42],    # neighborhood mean
        "horizon": 6,                      # predict the same slot on the next day
    }
    return config


def build_feature_names(n_types: int, time_feature_cols: list, config: dict):
    feature_names = []

    # Apply each feature group to all crime types
    for lag in config["lags"]:
        for c in range(n_types):
            feature_names.append(f"type{c}_lag_{lag}")

    for w in config["rolling_windows"]:
        for c in range(n_types):
            feature_names.append(f"type{c}_roll_mean_{w}")

    for w in config["rolling_std_windows"]:
        for c in range(n_types):
            feature_names.append(f"type{c}_roll_std_{w}")

    for w in config["neighbor_windows"]:
        for c in range(n_types):
            feature_names.append(f"type{c}_neighbor_mean_{w}")

    for col in time_feature_cols:
        feature_names.append(col)

    return feature_names


# =========================================================
# 3. Build features for all grids at a single anchor time
# =========================================================

def safe_window_mean(arr: np.ndarray, start: int, end: int):
    """
    arr: shape (T, G, C)
    Return: shape (G, C)
    """
    if end <= start:
        return np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float32)
    return arr[start:end].mean(axis=0, dtype=np.float32)


def safe_window_std(arr: np.ndarray, start: int, end: int):
    if end <= start:
        return np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float32)
    return arr[start:end].std(axis=0, dtype=np.float32)


def compute_neighbor_mean(values_gc: np.ndarray, neighbor_list: list):
    """
    values_gc: shape (G, C)
    Return: shape (G, C)
    """
    g_count, c_count = values_gc.shape
    out = np.zeros((g_count, c_count), dtype=np.float32)

    for g in range(g_count):
        nbs = neighbor_list[g]
        if len(nbs) == 0:
            continue
        out[g] = values_gc[nbs].mean(axis=0)
    return out


def compute_neighbor_mean_window(arr_tgc: np.ndarray, start: int, end: int, neighbor_list: list):
    """
    First compute the temporal mean over the window, then compute the neighborhood mean.
    """
    window_mean = safe_window_mean(arr_tgc, start, end)  # (G, C)
    return compute_neighbor_mean(window_mean, neighbor_list)


def build_features_for_time(
    tensor: np.ndarray,
    time_features: np.ndarray,
    t: int,
    neighbor_list: list,
    config: dict
):
    """
    Build the feature matrix for all grids at anchor time t.
    Output shape = (G, F)

    Here, t is the current anchor time, and the target is the crime count at t + horizon.
    """
    g_count = tensor.shape[1]
    n_types = tensor.shape[2]
    parts = []

    # 1) Lag features
    for lag in config["lags"]:
        idx = max(0, t - lag)
        parts.append(tensor[idx].astype(np.float32))  # (G, C)

    # 2) Rolling mean features
    for w in config["rolling_windows"]:
        parts.append(safe_window_mean(tensor, max(0, t - w), t))

    # 3) Rolling std features
    for w in config["rolling_std_windows"]:
        parts.append(safe_window_std(tensor, max(0, t - w), t))

    # 4) Neighborhood mean features
    for w in config["neighbor_windows"]:
        if w == 1:
            base = tensor[max(0, t - 1)].astype(np.float32)  # (G, C)
            parts.append(compute_neighbor_mean(base, neighbor_list))
        else:
            parts.append(compute_neighbor_mean_window(tensor, max(0, t - w), t, neighbor_list))

    # Concatenate all crime-type-related features
    x_gc = np.concatenate(parts, axis=1)  # (G, feature_groups * C)

    # 5) Time features
    tf = time_features[t].astype(np.float32)           # (T_feat,)
    tf_rep = np.repeat(tf.reshape(1, -1), g_count, axis=0)

    x = np.concatenate([x_gc, tf_rep], axis=1).astype(np.float32)
    return x


# =========================================================
# 4. Build split datasets
# =========================================================

def get_anchor_ranges(meta: dict, horizon: int):
    """
    Anchor time t is used to predict t + horizon.
    Therefore, each split must end at split_end - horizon.
    """
    lookback = meta["lookback"]
    train_end = meta["train_end"]
    val_end = meta["val_end"]
    n_steps = meta["n_steps"]

    # train: [lookback, train_end - horizon)
    train_t0 = lookback
    train_t1 = train_end - horizon

    # val: [train_end, val_end - horizon)
    val_t0 = train_end
    val_t1 = val_end - horizon

    # test: [val_end, n_steps - horizon)
    test_t0 = val_end
    test_t1 = n_steps - horizon

    return {
        "train": (train_t0, train_t1),
        "val": (val_t0, val_t1),
        "test": (test_t0, test_t1),
    }


def build_split_arrays(
    tensor: np.ndarray,
    time_features: np.ndarray,
    meta: dict,
    neighbor_list: list,
    split_name: str,
    output_dir: Path,
    config: dict
):
    """
    Build X and y for one split and save them as npy files.
    """
    ranges = get_anchor_ranges(meta, config["horizon"])
    t0, t1 = ranges[split_name]

    g_count = meta["n_grids"]
    n_types = meta["n_types"]
    time_feature_cols = meta["time_feature_cols"]

    feature_names = build_feature_names(n_types, time_feature_cols, config)
    n_features = len(feature_names)
    n_rows = (t1 - t0) * g_count

    print(f"\n[{split_name}] anchor range: [{t0}, {t1})")
    print(f"[{split_name}] rows = {n_rows:,}, n_features = {n_features}")

    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"

    # Use memmap to avoid loading everything into memory at once
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(n_rows, n_features)
    )
    y = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.float32, shape=(n_rows, n_types)
    )

    row_ptr = 0
    for t in tqdm(range(t0, t1), desc=f"Building {split_name}"):
        X_t = build_features_for_time(
            tensor=tensor,
            time_features=time_features,
            t=t,
            neighbor_list=neighbor_list,
            config=config
        )  # (G, F)

        y_t = tensor[t + config["horizon"]].astype(np.float32)  # (G, C)

        next_ptr = row_ptr + g_count
        X[row_ptr:next_ptr] = X_t
        y[row_ptr:next_ptr] = y_t
        row_ptr = next_ptr

    del X
    del y

    # Save feature names separately
    feature_df = pd.DataFrame({"feature_name": feature_names})
    feature_df.to_csv(output_dir / "feature_names.csv", index=False, encoding="utf-8-sig")

    return x_path, y_path, feature_names


# =========================================================
# 5. Training / evaluation
# =========================================================

def get_crime_type_names(meta: dict):
    # Use type_names from meta if available
    if "type_names" in meta:
        return meta["type_names"]

    # Default names for the current 5 crime categories
    default_names = [
        "THEFT",
        "BATTERY",
        "CRIMINAL_DAMAGE",
        "ASSAULT",
        "DECEPTIVE_PRACTICE"
    ]
    if meta["n_types"] == len(default_names):
        return default_names

    return [f"type_{i}" for i in range(meta["n_types"])]


def train_one_crime_type(
    crime_idx: int,
    crime_name: str,
    X_train: np.ndarray,
    y_train_all: np.ndarray,
    X_val: np.ndarray,
    y_val_all: np.ndarray,
    X_test: np.ndarray,
    y_test_all: np.ndarray,
    feature_names: list,
    model_dir: Path,
    pred_dir: Path,
    importance_dir: Path,
    random_state: int = 42
):
    print(f"\n==============================")
    print(f"Training crime type: {crime_name} (idx={crime_idx})")
    print(f"==============================")

    y_train = np.log1p(y_train_all[:, crime_idx])
    y_val = np.log1p(y_val_all[:, crime_idx])
    y_test = y_test_all[:, crime_idx].astype(np.float32)

    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=30,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Predict in log space first, then transform back
    pred_val_log = model.predict(X_val)
    pred_test_log = model.predict(X_test)

    pred_val = np.expm1(pred_val_log)
    pred_test = np.expm1(pred_test_log)

    pred_val = np.clip(pred_val, 0, None)
    pred_test = np.clip(pred_test, 0, None)

    true_val = y_val_all[:, crime_idx].astype(np.float32)
    true_test = y_test.astype(np.float32)

    metrics = {
        "crime_type": crime_name,
        "val_mae": float(mean_absolute_error(true_val, pred_val)),
        "val_rmse": float(rmse(true_val, pred_val)),
        "test_mae": float(mean_absolute_error(true_test, pred_test)),
        "test_rmse": float(rmse(true_test, pred_test)),
        "best_iteration": int(model.best_iteration) if model.best_iteration is not None else -1,
    }

    print(f"[{crime_name}] VAL  MAE={metrics['val_mae']:.4f} RMSE={metrics['val_rmse']:.4f}")
    print(f"[{crime_name}] TEST MAE={metrics['test_mae']:.4f} RMSE={metrics['test_rmse']:.4f}")

    # Save model
    model_path = model_dir / f"xgb_{crime_name}.json"
    model.save_model(model_path.as_posix())

    # Save predictions
    pred_df = pd.DataFrame({
        "y_true": true_test,
        "y_pred": pred_test,
    })
    pred_df.to_csv(pred_dir / f"test_predictions_{crime_name}.csv", index=False, encoding="utf-8-sig")

    # Save feature importance
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature_name": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(importance_dir / f"feature_importance_{crime_name}.csv", index=False, encoding="utf-8-sig")

    return metrics


# =========================================================
# 6. Main pipeline
# =========================================================

def main(args):
    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "arrays")
    ensure_dir(output_dir / "models")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "feature_importance")

    print("Loading data...")
    tensor = np.load(processed_dir / "tensor.npy", mmap_mode="r")          # (T, G, C)
    time_features = np.load(processed_dir / "time_features.npy")           # (T, 8)
    meta = load_meta(processed_dir)
    print("META KEYS:")
    print(meta.keys())

    print(f"tensor shape        : {tensor.shape}")
    print(f"time_features shape : {time_features.shape}")
    print(f"n_grids             : {meta['n_grids']}")
    print(f"n_types             : {meta['n_types']}")
    print(f"n_steps             : {meta['n_steps']}")
    print(f"lookback            : {meta['lookback']}")
    print(f"train_end           : {meta['train_end']}")
    print(f"val_end             : {meta['val_end']}")

    config = get_feature_config()

    # Check whether lookback is large enough to cover the maximum lag/window
    max_required = max(
        max(config["lags"]),
        max(config["rolling_windows"]),
        max(config["rolling_std_windows"]),
        max(config["neighbor_windows"])
    )
    if meta["lookback"] < max_required:
        raise ValueError(
            f"meta['lookback']={meta['lookback']} is not large enough "
            f"to support the maximum required window {max_required}"
        )

    neighbor_list = build_neighbor_list(meta)

    # 1) Build train/val/test arrays
    arrays_dir = output_dir / "arrays"

    x_train_path, y_train_path, feature_names = build_split_arrays(
        tensor=tensor,
        time_features=time_features,
        meta=meta,
        neighbor_list=neighbor_list,
        split_name="train",
        output_dir=arrays_dir,
        config=config
    )

    x_val_path, y_val_path, _ = build_split_arrays(
        tensor=tensor,
        time_features=time_features,
        meta=meta,
        neighbor_list=neighbor_list,
        split_name="val",
        output_dir=arrays_dir,
        config=config
    )

    x_test_path, y_test_path, _ = build_split_arrays(
        tensor=tensor,
        time_features=time_features,
        meta=meta,
        neighbor_list=neighbor_list,
        split_name="test",
        output_dir=arrays_dir,
        config=config
    )

    # 2) Load memmap arrays
    X_train = np.load(x_train_path, mmap_mode="r")
    y_train_all = np.load(y_train_path, mmap_mode="r")
    X_val = np.load(x_val_path, mmap_mode="r")
    y_val_all = np.load(y_val_path, mmap_mode="r")
    X_test = np.load(x_test_path, mmap_mode="r")
    y_test_all = np.load(y_test_path, mmap_mode="r")

    crime_names = get_crime_type_names(meta)
    if len(crime_names) != meta["n_types"]:
        crime_names = [f"type_{i}" for i in range(meta["n_types"])]

    # 3) Train one model for each crime type
    all_metrics = []
    for crime_idx, crime_name in enumerate(crime_names):
        metrics = train_one_crime_type(
            crime_idx=crime_idx,
            crime_name=crime_name,
            X_train=X_train,
            y_train_all=y_train_all,
            X_val=X_val,
            y_val_all=y_val_all,
            X_test=X_test,
            y_test_all=y_test_all,
            feature_names=feature_names,
            model_dir=output_dir / "models",
            pred_dir=output_dir / "predictions",
            importance_dir=output_dir / "feature_importance",
            random_state=args.random_state
        )
        all_metrics.append(metrics)

    # 4) Save overall metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "metrics_by_crime_type.csv", index=False, encoding="utf-8-sig")

    overall = {
        "avg_val_mae": float(metrics_df["val_mae"].mean()),
        "avg_val_rmse": float(metrics_df["val_rmse"].mean()),
        "avg_test_mae": float(metrics_df["test_mae"].mean()),
        "avg_test_rmse": float(metrics_df["test_rmse"].mean()),
    }

    with open(output_dir / "metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("Training finished.")
    print("==============================")
    print(metrics_df)
    print("\nOverall:")
    print(overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root data directory, e.g. ./data or ./数据v2"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./xgb_outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()
    main(args)