"""
utils.py — 各模型共用的特征工程工具
用法：from utils import get_neighbors, build_lag_features, CrimeDataset
"""

import numpy as np
import json
from pathlib import Path


def load_meta(out_dir="processed"):
    with open(Path(out_dir) / "meta.json") as f:
        meta = json.load(f)
    return meta


# ── 空间邻域 ──────────────────────────────────────────────────────────────────

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


def build_neighbor_matrix(meta: dict) -> list[list[int]]:
    """
    预计算所有网格的邻居索引，返回 list[list[int]]，长度 = N_GRIDS。
    """
    return [get_neighbors(g, meta) for g in range(meta["n_grids"])]


# ── 滞后特征（针对 XGBoost ） ──────────────────────────────────────

def build_lag_features(tensor: np.ndarray, t: int, g: int,
                       neighbor_list: list[list[int]]) -> np.ndarray:
    """
    为单个 (时间步 t, 网格 g) 构建平铺特征向量，供树模型使用。

    包含：
      - 滞后计数：t-1, t-6 (前一天同槽), t-42 (前一周同槽)
      - 7步滚动均值 / 标准差（约前1天）
      - 42步滚动均值（约前1周）
      - 邻域上一步均值

    参数
    ----
    tensor       : (N_STEPS, N_GRIDS, N_TYPES) int16
    t            : 目标时间步（要预测的步）
    g            : 目标网格索引
    neighbor_list: build_neighbor_matrix() 的输出

    返回
    ----
    feats : 1-D float32 数组，长度 = N_TYPES * (3滞后 + 2滚动 + 1邻域均值 + 1邻域7步均值)
    """
    N_TYPES = tensor.shape[2]
    feats = []

    for lag in [1, 6, 42]:
        idx = max(0, t - lag)
        feats.append(tensor[idx, g, :].astype(np.float32))

    # 7步滚动
    window7  = tensor[max(0, t-7):t,  g, :].astype(np.float32)
    window42 = tensor[max(0, t-42):t, g, :].astype(np.float32)
    feats.append(window7.mean(axis=0)  if len(window7)  else np.zeros(N_TYPES, np.float32))
    feats.append(window7.std(axis=0)   if len(window7)  else np.zeros(N_TYPES, np.float32))
    feats.append(window42.mean(axis=0) if len(window42) else np.zeros(N_TYPES, np.float32))

    # 邻域特征
    nbs = neighbor_list[g]
    if nbs:
        nb_prev   = tensor[max(0, t-1), nbs, :].astype(np.float32)
        nb_7      = tensor[max(0, t-7):t, :, :][:, nbs, :].astype(np.float32)
        feats.append(nb_prev.mean(axis=0))
        feats.append(nb_7.mean(axis=(0, 1)) if len(nb_7) else np.zeros(N_TYPES, np.float32))
    else:
        feats.append(np.zeros(N_TYPES, np.float32))
        feats.append(np.zeros(N_TYPES, np.float32))

    return np.concatenate(feats)


# ── PyTorch Dataset（lstm、st-gcn使用） ─────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset

    class CrimeDataset(Dataset):
        """
        滑动窗口 Dataset，避免预先展开整个 X（节省内存）。

        输出：
          x_seq   : (lookback, N_GRIDS, N_TYPES)  float32，log(count+1) 变换后
          x_time  : (lookback, 8)                 float32，时间特征
          y       : (N_GRIDS, N_TYPES)             float32，log(count+1)
        """

        def __init__(self, tensor: np.ndarray, time_features: np.ndarray,
                     start: int, end: int, lookback: int):
            """
            参数
            ----
            tensor        : (N_STEPS, N_GRIDS, N_TYPES)
            time_features : (N_STEPS, 8)
            start         : 第一个可预测时间步（含）= lookback 或 train_end-lookback
            end           : 最后一个可预测时间步（不含）
            lookback      : 历史窗口长度
            """
            self.tensor    = tensor.astype(np.float32)
            self.tf        = time_features
            self.lookback  = lookback
            # 可预测步范围：[start, end)，要求 start >= lookback
            self.indices   = np.arange(max(start, lookback), end)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            t       = self.indices[i]
            x_seq   = np.log1p(self.tensor[t - self.lookback : t])   # (L, G, C)
            x_time  = self.tf[t - self.lookback : t]                 # (L, 8)
            y       = np.log1p(self.tensor[t])                       # (G, C)
            return (torch.from_numpy(x_seq),
                    torch.from_numpy(x_time),
                    torch.from_numpy(y))

except ImportError:
    pass   # torch 未安装时跳过，不影响其他功能


# ── 快速加载入口 ──────────────────────────────────────────────────────────────

def load_data(out_dir="processed"):
    """
    返回完整 tensor, time_features, meta（需自行按 meta 中的边界切分）。
    """
    out_dir = Path(out_dir)
    tensor        = np.load(out_dir / "tensor.npy")
    time_features = np.load(out_dir / "time_features.npy")
    meta          = load_meta(out_dir)
    return tensor, time_features, meta
