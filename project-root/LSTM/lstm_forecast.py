# lstm_grid_forecast.py
import os
import json
import random
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "processed"

TENSOR_PATH = DATA_DIR / "tensor.npy"
TIME_FEATURES_PATH = DATA_DIR / "time_features.npy"
META_PATH = DATA_DIR / "meta.json"

LOOKBACK = 180          # 30 days * 6 slots/day = 180
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

MODEL_SAVE_PATH = "best_lstm_grid_total_with_neighbors.pt"
PRED_SAVE_DIR = "lstm_preds_with_neighbors"

# neighbor region feature
USE_NEIGHBOR_SUM = True
USE_NEIGHBOR_MEAN = True

# reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# spatial neighbor utils
def get_neighbors(grid_idx: int, meta: dict) -> list[int]:
    """
    返回 grid_idx 对应格子的 8-连通邻居索引列表，越界自动跳过。
    grid_idx 即 grid_id = row * N_COLS + col，张量第二维直接对应。
    """
    N_ROWS = meta["n_rows"]
    N_COLS = meta["n_cols"]
    row, col = grid_idx // N_COLS, grid_idx % N_COLS

    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < N_ROWS and 0 <= nc < N_COLS:
                neighbors.append(nr * N_COLS + nc)
    return neighbors


def build_neighbor_map(meta: dict, n_grids: int):
    neighbor_map = []
    for g in range(n_grids):
        neighbor_map.append(get_neighbors(g, meta))
    return neighbor_map


def build_spatial_features(total_counts: np.ndarray, meta: dict) -> np.ndarray:
    """

    according to sum of total crime number，construct the space neighbour's input feature

    total_counts: (T, G)
    return:
        spatial_x: (T, D_spatial)

    """
    T, G = total_counts.shape
    neighbor_map = build_neighbor_map(meta, G)

    own_feat = total_counts.astype(np.float32)  # (T, G)

    feature_list = [own_feat]

    if USE_NEIGHBOR_SUM:
        neighbor_sum = np.zeros((T, G), dtype=np.float32)
        for g in range(G):
            nbrs = neighbor_map[g]
            if len(nbrs) > 0:
                neighbor_sum[:, g] = total_counts[:, nbrs].sum(axis=1)
        feature_list.append(neighbor_sum)

    if USE_NEIGHBOR_MEAN:
        neighbor_mean = np.zeros((T, G), dtype=np.float32)
        for g in range(G):
            nbrs = neighbor_map[g]
            if len(nbrs) > 0:
                neighbor_mean[:, g] = total_counts[:, nbrs].mean(axis=1)
        feature_list.append(neighbor_mean)

    spatial_x = np.concatenate(feature_list, axis=1)  # (T, G * num_feature_groups)
    return spatial_x


# Dataset
class CrimeSeqDataset(Dataset):
    """
    每个样本 = 过去 LOOKBACK 步 -> 预测当前 t 这一步（即未来4h）
    X: (lookback, input_dim)
    y: (n_grids,)
    """
    def __init__(self, x_all, y_all, start, end, lookback):
        """
        x_all: (T, input_dim)
        y_all: (T, n_grids)
        start/end: target t 的范围 [start, end)
        """
        self.x_all = x_all.astype(np.float32)
        self.y_all = y_all.astype(np.float32)
        self.lookback = lookback
        self.indices = np.arange(max(start, lookback), end)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.x_all[t - self.lookback:t]   # (L, D)
        y = self.y_all[t]                     # (G,)
        return torch.from_numpy(x), torch.from_numpy(y)

# Model
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
        # x: (B, L, D)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (B, H)
        pred = self.head(last_hidden)  # (B, G)
        return pred


# Metrics
def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def occurrence_acc_np(y_true, y_pred, threshold=0.0):
    """
    用于回归任务的 ACC：
    将预测和真实值转为“是否发生犯罪”的二分类，再算准确率
    """
    true_bin = (y_true > threshold).astype(np.int32)
    pred_bin = (y_pred > threshold).astype(np.int32)
    return np.mean(true_bin == pred_bin)


# Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
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

        # 转回原始计数空间评估
        pred = torch.expm1(pred_log).clamp(min=0.0)
        true = torch.expm1(y).clamp(min=0.0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(true.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)     # (N, G)
    all_targets = np.concatenate(all_targets, axis=0) # (N, G)

    mae = mae_np(all_targets, all_preds)
    rmse = rmse_np(all_targets, all_preds)
    acc = occurrence_acc_np(all_targets, all_preds, threshold=0.0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": mae,
        "rmse": rmse,
        "acc": acc,
        "preds": all_preds,
        "targets": all_targets,
    }

# Main
def main():
    seed_everything(SEED)
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load data ----
    tensor = np.load(TENSOR_PATH)                # (T, G, C)
    time_features = np.load(TIME_FEATURES_PATH)  # (T, 8)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    T, G, C = tensor.shape
    train_end = meta["train_end"]
    val_end = meta["val_end"]

    print(f"tensor shape = {tensor.shape}")
    print(f"time_features shape = {time_features.shape}")
    print(f"train_end = {train_end}, val_end = {val_end}, total_steps = {T}")
    print(f"n_grids = {G}, n_types = {C}")

    if LOOKBACK != meta["lookback"]:
        print(f"[WARN] Config LOOKBACK={LOOKBACK}, but meta lookback={meta['lookback']}")
        
    # 目标：预测每个网格“未来4h总犯罪数”
    # 做法：
    # 1) 先把 5 类犯罪按网格求和 => (T, G)
    # 2) 对输入加入空间邻域特征：
    #    - own count
    #    - neighbor sum
    #    - neighbor mean
    # 3) 再拼接时间特征
    # 输出：当前 t 这一步所有网格的总数
    total_counts = tensor.sum(axis=2).astype(np.float32)   # (T, G)

    # 构造带空间信息的输入
    spatial_x = build_spatial_features(total_counts, meta)  # (T, D_spatial)

    # log1p 稳定训练
    x_spatial = np.log1p(spatial_x)                         # (T, D_spatial)
    y_all = np.log1p(total_counts)                          # (T, G)

    # 拼接时间特征
    x_all = np.concatenate([x_spatial, time_features.astype(np.float32)], axis=1)

    # 只用训练段统计量做标准化，避免泄漏
    train_x = x_all[:train_end]
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_all = (x_all - x_mean) / x_std

    print(f"spatial feature dim = {x_spatial.shape[1]}")
    print(f"time feature dim    = {time_features.shape[1]}")
    print(f"input_dim           = {x_all.shape[1]}")
    print(f"output_dim          = {G}")

    # ---- dataset split ----
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

    # ---- model ----
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

    # ---- train ----
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
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_MAE={val_metrics['mae']:.6f} | "
            f"val_RMSE={val_metrics['rmse']:.6f} | "
            f"val_ACC={val_metrics['acc']:.6f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, MODEL_SAVE_PATH)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ---- load best ----
    if best_state is None:
        best_state = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(best_state)

    # ---- final eval ----
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print("\n===== Final Results =====")
    print(f"Validation MAE  : {val_metrics['mae']:.6f}")
    print(f"Validation RMSE : {val_metrics['rmse']:.6f}")
    print(f"Validation ACC  : {val_metrics['acc']:.6f}")
    print(f"Test MAE        : {test_metrics['mae']:.6f}")
    print(f"Test RMSE       : {test_metrics['rmse']:.6f}")
    print(f"Test ACC        : {test_metrics['acc']:.6f}")

    # 保存预测
    np.save(os.path.join(PRED_SAVE_DIR, "val_preds.npy"), val_metrics["preds"])
    np.save(os.path.join(PRED_SAVE_DIR, "val_targets.npy"), val_metrics["targets"])
    np.save(os.path.join(PRED_SAVE_DIR, "test_preds.npy"), test_metrics["preds"])
    np.save(os.path.join(PRED_SAVE_DIR, "test_targets.npy"), test_metrics["targets"])

    # 保存指标
    metrics = {
        "val_mae": float(val_metrics["mae"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_acc": float(val_metrics["acc"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_acc": float(test_metrics["acc"]),
    }
    with open(os.path.join(PRED_SAVE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Predictions saved to: {PRED_SAVE_DIR}/")
    print(f"Metrics saved to: {os.path.join(PRED_SAVE_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
