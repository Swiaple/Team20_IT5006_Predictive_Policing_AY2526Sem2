#!/usr/bin/env python3
"""
Prepare STGCN-ready arrays from dataset/v2.

Outputs:
  - tensor.npy          shape: (n_steps, n_grids, n_types)
  - time_features.npy   shape: (n_steps, 8)
  - adj.npz             sparse adjacency with 8-neighborhood + self-loop
  - meta.json           copied from dataset/v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


TIME_FEATURE_COLS = [
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]


def load_meta(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_tensor(dataset_dir: Path, meta: dict) -> np.ndarray:
    n_steps = int(meta["n_steps"])
    n_grids = int(meta["n_grids"])
    n_types = int(meta["n_types"])
    tensor = np.zeros((n_steps, n_grids, n_types), dtype=np.int16)

    files = ["events_train.parquet", "events_val.parquet", "events_test.parquet"]
    for fn in files:
        df = pd.read_parquet(dataset_dir / fn, columns=["time_step", "grid_id", "crime_type"])
        ts = df["time_step"].to_numpy(np.int64)
        gs = df["grid_id"].to_numpy(np.int64)
        cs = df["crime_type"].to_numpy(np.int64)

        valid = (
            (ts >= 0)
            & (ts < n_steps)
            & (gs >= 0)
            & (gs < n_grids)
            & (cs >= 0)
            & (cs < n_types)
        )
        if not np.all(valid):
            ts = ts[valid]
            gs = gs[valid]
            cs = cs[valid]

        np.add.at(tensor, (ts, gs, cs), 1)

    return tensor


def build_day_index(meta: dict) -> list[pd.Timestamp]:
    n_days = int(meta["n_steps"]) // 6
    idx_to_date = [pd.Timestamp("1970-01-01")] * n_days
    for date_str, idx in meta["date_to_idx"].items():
        i = int(idx)
        if 0 <= i < n_days:
            idx_to_date[i] = pd.Timestamp(date_str)
    return idx_to_date


def build_day_holiday_map(dataset_dir: Path, meta: dict) -> np.ndarray:
    n_days = int(meta["n_steps"]) // 6
    day_holiday = np.full(n_days, -1, dtype=np.int8)

    files = ["events_train.parquet", "events_val.parquet", "events_test.parquet"]
    for fn in files:
        df = pd.read_parquet(dataset_dir / fn, columns=["time_step", "is_holiday"])
        day_idx = (df["time_step"].to_numpy(np.int64) // 6).astype(np.int64)
        is_holiday = df["is_holiday"].to_numpy(np.int8)
        for d, h in zip(day_idx, is_holiday):
            if 0 <= d < n_days and day_holiday[d] < 0:
                day_holiday[d] = h

    day_holiday[day_holiday < 0] = 0
    return day_holiday


def build_time_features(meta: dict, day_holiday: np.ndarray, idx_to_date: list[pd.Timestamp]) -> np.ndarray:
    n_steps = int(meta["n_steps"])
    slots_per_day = 6
    tf = np.zeros((n_steps, 8), dtype=np.float32)

    for t in range(n_steps):
        slot = t % slots_per_day
        day_idx = t // slots_per_day
        date = idx_to_date[day_idx]
        weekday = date.dayofweek
        month = int(date.month)

        tf[t, 0] = np.sin(2 * np.pi * slot / slots_per_day)
        tf[t, 1] = np.cos(2 * np.pi * slot / slots_per_day)
        tf[t, 2] = np.sin(2 * np.pi * weekday / 7)
        tf[t, 3] = np.cos(2 * np.pi * weekday / 7)
        tf[t, 4] = np.sin(2 * np.pi * month / 12)
        tf[t, 5] = np.cos(2 * np.pi * month / 12)
        tf[t, 6] = float(weekday >= 5)
        tf[t, 7] = float(day_holiday[day_idx])

    return tf


def build_adjacency(meta: dict) -> sp.csr_matrix:
    n_grids = int(meta["n_grids"])

    rows = []
    cols = []
    data = []
    for g in range(n_grids):
        for nb in get_neighbors(g, meta):
            if 0 <= nb < n_grids:
                rows.append(g)
                cols.append(nb)
                data.append(1.0)

    adj = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_grids, n_grids),
        dtype=np.float32,
    )
    return adj.tocsr()


def get_neighbors(grid_idx: int, meta: dict) -> list[int]:
    """
    Return 8-connected neighbor indices for the given grid_idx.
    This follows the user's specified logic and excludes itself.
    """
    n_rows = int(meta["n_rows"])
    n_cols = int(meta["n_cols"])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset/v2",
        help="Path to dataset/v2 directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/data_v2",
        help="Output directory for prepared arrays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(dataset_dir)

    print("Building tensor...")
    tensor = build_tensor(dataset_dir, meta)
    print(f"tensor shape: {tensor.shape}, non-zero: {np.count_nonzero(tensor):,}")

    print("Building time features...")
    idx_to_date = build_day_index(meta)
    day_holiday = build_day_holiday_map(dataset_dir, meta)
    time_features = build_time_features(meta, day_holiday, idx_to_date)
    print(f"time_features shape: {time_features.shape}")

    print("Building adjacency...")
    adj = build_adjacency(meta)
    print(f"adj shape: {adj.shape}, nnz: {adj.nnz:,}")

    np.save(output_dir / "tensor.npy", tensor)
    np.save(output_dir / "time_features.npy", time_features)
    sp.save_npz(output_dir / "adj.npz", adj)

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"  {output_dir / 'tensor.npy'}")
    print(f"  {output_dir / 'time_features.npy'}")
    print(f"  {output_dir / 'adj.npz'}")
    print(f"  {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
