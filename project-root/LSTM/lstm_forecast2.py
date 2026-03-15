import os
import json
import math
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed"

TENSOR_PATH = DATA_DIR / "tensor.npy"
TIME_FEATURES_PATH = DATA_DIR / "time_features.npy"
META_PATH = DATA_DIR / "meta.json"

LOOKBACK = 180
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 20
PATIENCE = 5
SEED = 42

NUM_WORKERS = 0
PIN_MEMORY = True

USE_NEIGHBOR_SUM = True
USE_NEIGHBOR_MEAN = True

MODEL_DIR = BASE_DIR / "models_by_type"
PRED_DIR = BASE_DIR / "lstm_preds_by_type"

DEFAULT_TYPE_NAMES = [
    "type_0",
    "type_1",
    "type_2",
    "type_3",
    "type_4",
]


# =========================
# Reproducibility
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Spatial Utils
# =========================
def get_neighbors(grid_idx: int, meta: dict) -> list[int]:
    n_rows = meta["n_rows"]
    n_cols = meta["n_cols"]
    row, col = divmod(grid_idx, n_cols)

    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                neighbors.append(nr * n_cols + nc)
    return neighbors


def build_neighbor_map(meta: dict, n_grids: int):
    return [get_neighbors(g, meta) for g in range(n_grids)]


def build_spatial_features(single_type_counts: np.ndarray, meta: dict) -> np.ndarray:
    """
    single_type_counts: (T, G)
    return: (T, G * k)
    """
    T, G = single_type_counts.shape
    neighbor_map = build_neighbor_map(meta, G)

    own_feat = single_type_counts.astype(np.float32)
    feature_list = [own_feat]

    if USE_NEIGHBOR_SUM:
        neighbor_sum = np.zeros((T, G), dtype=np.float32)
        for g in range(G):
            nbrs = neighbor_map[g]
            if nbrs:
                neighbor_sum[:, g] = single_type_counts[:, nbrs].sum(axis=1)
        feature_list.append(neighbor_sum)

    if USE_NEIGHBOR_MEAN:
        neighbor_mean = np.zeros((T, G), dtype=np.float32)
        for g in range(G):
            nbrs = neighbor_map[g]
            if nbrs:
                neighbor_mean[:, g] = single_type_counts[:, nbrs].mean(axis=1)
        feature_list.append(neighbor_mean)

    spatial_x = np.concatenate(feature_list, axis=1)
    return spatial_x


# =========================
# Dataset
# =========================
class CrimeSeqDataset(Dataset):
    def __init__(self, x_all, y_all, start, end, lookback):
        self.x_all = x_all.astype(np.float32)
        self.y_all = y_all.astype(np.float32)
        self.lookback = lookback
        self.indices = np.arange(max(start, lookback), end)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.x_all[t - self.lookback:t]
        y = self.y_all[t]
        return torch.from_numpy(x), torch.from_numpy(y)


# =========================
# Model
# =========================
class LSTMGridRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        pred = self.head(last_hidden)
        return pred


# =========================
# Basic Metrics
# =========================
def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def occurrence_acc_np(y_true, y_pred, threshold=0.0):
    true_bin = (y_true > threshold).astype(np.int32)
    pred_bin = (y_pred > threshold).astype(np.int32)
    return float(np.mean(true_bin == pred_bin))


# =========================
# Temporal Metric
# =========================
def compute_temporal_precision(y_true: np.ndarray, y_pred: np.ndarray, temporal_quantile: float):
    """
    y_true, y_pred: (N, G)
    先按城市总量聚合，再用 quantile 阈值划分高风险时段
    """
    true_city = y_true.sum(axis=1)
    pred_city = y_pred.sum(axis=1)

    true_thr = float(np.quantile(true_city, temporal_quantile))
    pred_thr = float(np.quantile(pred_city, temporal_quantile))

    true_high = true_city >= true_thr
    pred_high = pred_city >= pred_thr

    tp = int(np.sum(true_high & pred_high))
    pred_pos = int(np.sum(pred_high))
    true_pos = int(np.sum(true_high))

    precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall = float(tp / true_pos) if true_pos > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = float(np.mean(true_high == pred_high))

    return {
        "temporal_quantile": float(temporal_quantile),
        "temporal_true_threshold": true_thr,
        "temporal_pred_threshold": pred_thr,
        "tp": tp,
        "pred_positive_count": pred_pos,
        "true_positive_count": true_pos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
    }


# =========================
# Spatial Metric
# =========================
def compute_spatial_metrics(y_true: np.ndarray, y_pred: np.ndarray, spatial_hotspot_ratio: float):
    """
    每个时段取预测前 top_k 网格作为热点，计算 PAI
    """
    N, G = y_true.shape
    top_k = int(math.ceil(G * spatial_hotspot_ratio))
    top_k = max(1, min(top_k, G))

    pai_list = []
    coverage_list = []
    hit_rate_list = []

    for t in range(N):
        true_t = y_true[t]
        pred_t = y_pred[t]

        pred_hot_idx = np.argpartition(pred_t, -top_k)[-top_k:]
        pred_hot_mask = np.zeros(G, dtype=bool)
        pred_hot_mask[pred_hot_idx] = True

        true_hot_idx = np.argpartition(true_t, -top_k)[-top_k:]
        true_hot_mask = np.zeros(G, dtype=bool)
        true_hot_mask[true_hot_idx] = True

        crimes_in_pred_hotspots = float(true_t[pred_hot_mask].sum())
        total_crimes = float(true_t.sum())

        area_ratio = top_k / G
        coverage = float(crimes_in_pred_hotspots / total_crimes) if total_crimes > 0 else 0.0
        pai = float(coverage / area_ratio) if area_ratio > 0 else 0.0
        hit_rate = float(np.mean(pred_hot_mask[true_hot_mask]))

        pai_list.append(pai)
        coverage_list.append(coverage)
        hit_rate_list.append(hit_rate)

    return {
        "spatial_hotspot_ratio": float(spatial_hotspot_ratio),
        "top_k": int(top_k),
        "mean_pai": float(np.mean(pai_list)),
        "std_pai": float(np.std(pai_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "std_coverage": float(np.std(coverage_list)),
        "mean_hit_rate": float(np.mean(hit_rate_list)),
        "std_hit_rate": float(np.std(hit_rate_list)),
    }


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, loader, device, temporal_quantile=0.8, spatial_hotspot_ratio=0.05):
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    criterion = nn.MSELoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred_log = model(x)
        loss = criterion(pred_log, y)

        total_loss += loss.item()
        n_batches += 1

        pred = torch.expm1(pred_log).clamp(min=0.0)
        true = torch.expm1(y).clamp(min=0.0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(true.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = mae_np(all_targets, all_preds)
    rmse = rmse_np(all_targets, all_preds)
    acc = occurrence_acc_np(all_targets, all_preds, threshold=0.0)

    temporal_metrics = compute_temporal_precision(
        y_true=all_targets,
        y_pred=all_preds,
        temporal_quantile=temporal_quantile,
    )
    spatial_metrics = compute_spatial_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        spatial_hotspot_ratio=spatial_hotspot_ratio,
    )

    return {
        "loss": float(total_loss / max(n_batches, 1)),
        "mae": mae,
        "rmse": rmse,
        "acc": acc,
        "preds": all_preds,
        "targets": all_targets,
        "temporal": temporal_metrics,
        "spatial": spatial_metrics,
    }


# =========================
# Load Data
# =========================
def load_data():
    tensor = np.load(TENSOR_PATH)                 # (T, G, C)
    time_features = np.load(TIME_FEATURES_PATH)   # (T, F)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "type_names" not in meta:
        c = tensor.shape[2]
        meta["type_names"] = DEFAULT_TYPE_NAMES[:c]

    return tensor, time_features, meta


# =========================
# Train One Type
# =========================
def train_one_type(
    type_idx: int,
    type_name: str,
    tensor: np.ndarray,
    time_features: np.ndarray,
    meta: dict,
    device: torch.device,
    temporal_quantile: float,
    spatial_hotspot_ratio: float,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    T, G, C = tensor.shape
    train_end = meta["train_end"]
    val_end = meta["val_end"]

    print("\n" + "=" * 80)
    print(f"Training type {type_idx}: {type_name}")
    print("=" * 80)

    single_type_counts = tensor[:, :, type_idx].astype(np.float32)  # (T, G)

    spatial_x = build_spatial_features(single_type_counts, meta)
    x_spatial = np.log1p(spatial_x)
    y_all = np.log1p(single_type_counts)

    x_all = np.concatenate([x_spatial, time_features.astype(np.float32)], axis=1)

    train_x = x_all[:train_end]
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_all = (x_all - x_mean) / x_std

    train_ds = CrimeSeqDataset(
        x_all=x_all,
        y_all=y_all,
        start=LOOKBACK,
        end=train_end,
        lookback=LOOKBACK,
    )
    val_ds = CrimeSeqDataset(
        x_all=x_all,
        y_all=y_all,
        start=train_end,
        end=val_end,
        lookback=LOOKBACK,
    )
    test_ds = CrimeSeqDataset(
        x_all=x_all,
        y_all=y_all,
        start=val_end,
        end=T,
        lookback=LOOKBACK,
    )

    print(f"train samples = {len(train_ds)}")
    print(f"val samples   = {len(val_ds)}")
    print(f"test samples  = {len(test_ds)}")
    print(f"input_dim     = {x_all.shape[1]}")
    print(f"output_dim    = {G}")
    print(f"top_k         = {math.ceil(G * spatial_hotspot_ratio)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = LSTMGridRegressor(
        input_dim=x_all.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=G,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    safe_name = str(type_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    model_path = MODEL_DIR / f"best_lstm_{safe_name}.pt"
    out_dir = PRED_DIR / safe_name
    os.makedirs(out_dir, exist_ok=True)

    best_val_mae = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_log = model(x)
            loss = criterion(pred_log, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            temporal_quantile=temporal_quantile,
            spatial_hotspot_ratio=spatial_hotspot_ratio,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_MAE={val_metrics['mae']:.6f} | "
            f"val_RMSE={val_metrics['rmse']:.6f} | "
            f"val_ACC={val_metrics['acc']:.6f} | "
            f"val_TemporalPrecision={val_metrics['temporal']['precision']:.6f} | "
            f"val_SpatialPAI={val_metrics['spatial']['mean_pai']:.6f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is None:
        best_state = torch.load(model_path, map_location=device)
    model.load_state_dict(best_state)

    val_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        temporal_quantile=temporal_quantile,
        spatial_hotspot_ratio=spatial_hotspot_ratio,
    )
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        temporal_quantile=temporal_quantile,
        spatial_hotspot_ratio=spatial_hotspot_ratio,
    )

    print("\n----- Final Results -----")
    print(f"[{type_name}] Validation MAE                : {val_metrics['mae']:.6f}")
    print(f"[{type_name}] Validation RMSE               : {val_metrics['rmse']:.6f}")
    print(f"[{type_name}] Validation ACC                : {val_metrics['acc']:.6f}")
    print(f"[{type_name}] Validation Temporal Precision : {val_metrics['temporal']['precision']:.6f}")
    print(f"[{type_name}] Validation Spatial PAI        : {val_metrics['spatial']['mean_pai']:.6f}")
    print(f"[{type_name}] Test MAE                      : {test_metrics['mae']:.6f}")
    print(f"[{type_name}] Test RMSE                     : {test_metrics['rmse']:.6f}")
    print(f"[{type_name}] Test ACC                      : {test_metrics['acc']:.6f}")
    print(f"[{type_name}] Test Temporal Precision       : {test_metrics['temporal']['precision']:.6f}")
    print(f"[{type_name}] Test Spatial PAI              : {test_metrics['spatial']['mean_pai']:.6f}")

    np.save(out_dir / "val_preds.npy", val_metrics["preds"])
    np.save(out_dir / "val_targets.npy", val_metrics["targets"])
    np.save(out_dir / "test_preds.npy", test_metrics["preds"])
    np.save(out_dir / "test_targets.npy", test_metrics["targets"])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "crime_type": type_name,
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_acc": float(val_metrics["acc"]),
            "test_mae": float(test_metrics["mae"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_acc": float(test_metrics["acc"]),
        }, f, indent=2, ensure_ascii=False)

    with open(out_dir / "metrics_temporal_val.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics["temporal"], f, indent=2, ensure_ascii=False)

    with open(out_dir / "metrics_temporal_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["temporal"], f, indent=2, ensure_ascii=False)

    with open(out_dir / "metrics_spatial_val.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics["spatial"], f, indent=2, ensure_ascii=False)

    with open(out_dir / "metrics_spatial_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["spatial"], f, indent=2, ensure_ascii=False)

    return {
        "crime_type": type_name,
        "model_path": str(model_path),
        "output_dir": str(out_dir),
        "val": {
            "mae": float(val_metrics["mae"]),
            "rmse": float(val_metrics["rmse"]),
            "acc": float(val_metrics["acc"]),
            "temporal_precision": float(val_metrics["temporal"]["precision"]),
            "temporal_true_threshold": float(val_metrics["temporal"]["temporal_true_threshold"]),
            "temporal_pred_threshold": float(val_metrics["temporal"]["temporal_pred_threshold"]),
            "spatial_pai": float(val_metrics["spatial"]["mean_pai"]),
            "top_k": int(val_metrics["spatial"]["top_k"]),
        },
        "test": {
            "mae": float(test_metrics["mae"]),
            "rmse": float(test_metrics["rmse"]),
            "acc": float(test_metrics["acc"]),
            "temporal_precision": float(test_metrics["temporal"]["precision"]),
            "temporal_true_threshold": float(test_metrics["temporal"]["temporal_true_threshold"]),
            "temporal_pred_threshold": float(test_metrics["temporal"]["temporal_pred_threshold"]),
            "spatial_pai": float(test_metrics["spatial"]["mean_pai"]),
            "top_k": int(test_metrics["spatial"]["top_k"]),
        },
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal_quantile", type=float, default=0.8)
    parser.add_argument("--spatial_hotspot_ratio", type=float, default=0.05)
    args = parser.parse_args()

    seed_everything(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tensor, time_features, meta = load_data()

    T, G, C = tensor.shape
    train_end = meta["train_end"]
    val_end = meta["val_end"]

    type_names = meta.get("type_names", DEFAULT_TYPE_NAMES[:C])
    if len(type_names) != C:
        type_names = [f"type_{i}" for i in range(C)]

    print(f"tensor shape = {tensor.shape}")
    print(f"time_features shape = {time_features.shape}")
    print(f"train_end = {train_end}, val_end = {val_end}, total_steps = {T}")
    print(f"n_grids = {G}, n_types = {C}")
    print(f"temporal_quantile = {args.temporal_quantile}")
    print(f"spatial_hotspot_ratio = {args.spatial_hotspot_ratio}")
    print(f"top_k = ceil({G} * {args.spatial_hotspot_ratio}) = {math.ceil(G * args.spatial_hotspot_ratio)}")

    if "lookback" in meta and LOOKBACK != meta["lookback"]:
        print(f"[WARN] Config LOOKBACK={LOOKBACK}, but meta lookback={meta['lookback']}")

    all_results = []

    for type_idx in range(C):
        result = train_one_type(
            type_idx=type_idx,
            type_name=type_names[type_idx],
            tensor=tensor,
            time_features=time_features,
            meta=meta,
            device=device,
            temporal_quantile=args.temporal_quantile,
            spatial_hotspot_ratio=args.spatial_hotspot_ratio,
        )
        all_results.append(result)

    with open(PRED_DIR / "summary_all_types.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nAll done.")
    print(f"Summary saved to: {PRED_DIR / 'summary_all_types.json'}")


if __name__ == "__main__":
    main()