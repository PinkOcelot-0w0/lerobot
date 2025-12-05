#!/usr/bin/env python
"""
修复数据集：从现有的数据和视频文件重新生成完整的 meta/episodes parquet 文件
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAT_NAMES = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


def _compute_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    stats = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=0),
        "count": np.array([arr.shape[0]]),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q10": np.quantile(arr, 0.10, axis=0),
        "q50": np.quantile(arr, 0.50, axis=0),
        "q90": np.quantile(arr, 0.90, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
    }
    return stats


def _to_obj_array(val):
    """确保统计值存为 ndarray(1D)，保持 pandas dtype=object，避免被推断为标量数值。"""
    arr = np.asarray(val)
    return arr.reshape(-1)


def _load_tasks(tasks_path: Path) -> dict[int, str]:
    if not tasks_path.exists():
        return {}
    df = pd.read_parquet(tasks_path)
    if "task_index" in df.columns:
        if "task" in df.columns:
            return dict(zip(df["task_index"], df["task"]))
        return {row.task_index: idx for idx, row in df.iterrows()}
    if df.index.name == "task":
        return {row.task_index: idx for idx, row in df.reset_index().iterrows()}
    return {}


def repair_dataset(dataset_root: str | Path) -> None:
    root = Path(dataset_root)
    info = json.loads((root / "meta/info.json").read_text())
    stats_global = {}
    stats_path = root / "meta/stats.json"
    if stats_path.exists():
        stats_global = json.loads(stats_path.read_text())

    tasks_map = _load_tasks(root / "meta/tasks.parquet")

    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    data_files = sorted((root / "data").glob("*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError("data/*.parquet 未找到")

    episodes_rows = []
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        chunk_idx = int(data_file.parent.name.split("-")[1])
        file_idx = int(data_file.stem.split("-")[1])

        for ep_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep_idx]

            length = len(ep_df)
            from_idx = int(ep_df["index"].min()) if "index" in ep_df else None
            to_idx = int(ep_df["index"].max()) + 1 if "index" in ep_df else None

            ts_min = float(ep_df["timestamp"].min()) if "timestamp" in ep_df else 0.0
            ts_max = float(ep_df["timestamp"].max()) if "timestamp" in ep_df else ts_min + length / info.get("fps", 30)

            # 任务文本
            if "task_index" in ep_df.columns:
                task_indices = ep_df["task_index"].unique().tolist()
            else:
                task_indices = []
            task_texts = [tasks_map.get(ti, str(ti)) for ti in task_indices]

            row = {}
            row["episode_index"] = int(ep_idx)
            row["tasks"] = task_texts
            row["length"] = int(length)
            row["data/chunk_index"] = chunk_idx
            row["data/file_index"] = file_idx
            row["dataset_from_index"] = from_idx if from_idx is not None else (len(episodes_rows) and episodes_rows[-1]["dataset_to_index"] or 0)
            row["dataset_to_index"] = to_idx if to_idx is not None else (from_idx or 0) + length

            # 统计列
            feature_stats_order = ["action", "observation.state", "observation.images.front", "observation.images.wrist"]
            for ft_key in feature_stats_order:
                if ft_key not in features:
                    continue
                ft = features[ft_key]
                dtype = ft.get("dtype")
                if dtype in ("image", "video"):
                    if ft_key in stats_global:
                        for stat_name in STAT_NAMES:
                            if stat_name in stats_global[ft_key]:
                                val = stats_global[ft_key][stat_name]
                                row[f"stats/{ft_key}/{stat_name}"] = _to_obj_array(val)
                else:
                    if ft_key in ep_df.columns:
                        vals = ep_df[ft_key].to_numpy()
                        if vals.dtype == object:
                            vals = np.stack([np.asarray(v) for v in vals])
                        stats = _compute_stats(vals)
                        for stat_name in STAT_NAMES:
                            row[f"stats/{ft_key}/{stat_name}"] = _to_obj_array(stats[stat_name])

            # metadata 统计
            metadata_stats_order = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
            for col in metadata_stats_order:
                if col in ep_df.columns:
                    stats = _compute_stats(ep_df[col].to_numpy())
                    for stat_name in STAT_NAMES:
                        row[f"stats/{col}/{stat_name}"] = _to_obj_array(stats[stat_name])

            # meta/episodes
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0

            # 视频元数据
            for vk in video_keys:
                row[f"videos/{vk}/chunk_index"] = chunk_idx
                row[f"videos/{vk}/file_index"] = file_idx
                row[f"videos/{vk}/from_timestamp"] = ts_min
                row[f"videos/{vk}/to_timestamp"] = ts_max

            episodes_rows.append(row)

    episodes_rows.sort(key=lambda r: r["dataset_from_index"])

    df_result = pd.DataFrame(episodes_rows)
    table = pa.Table.from_pandas(df_result, preserve_index=False)

    out_dir = root / "meta" / "episodes" / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "file-000.parquet"
    pq.write_table(table, out_path, compression="snappy", use_dictionary=True)

    logger.info("写入完成: %s", out_path)
    logger.info("总 episodes: %d", len(episodes_rows))
    logger.info("总列数: %d", len(table.column_names))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python repair_meta.py <dataset_root>")
        sys.exit(1)

    repair_dataset(sys.argv[1])
    print("完成")