import pandas as pd
from pathlib import Path
import numpy as np

def analyze_episodes_file(path: Path, label: str):
    """分析 episodes 文件的结构"""
    df = pd.read_parquet(path)
    
    print(f"\n{'='*60}")
    print(f"{label}: {path.name}")
    print(f"{'='*60}")
    print(f"总 episodes 数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"文件大小: {path.stat().st_size / 1024:.2f} KB")
    
    return df

def compare_columns(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    """对比两个 DataFrame 的列结构"""
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    order_match = list(df1.columns) == list(df2.columns)

    diffs = []
    if only_in_1:
        diffs.append(f"仅在 {label1} 中存在的列 ({len(only_in_1)}):\n  - " + "\n  - ".join(sorted(only_in_1)))
    if only_in_2:
        diffs.append(f"仅在 {label2} 中存在的列 ({len(only_in_2)}):\n  - " + "\n  - ".join(sorted(only_in_2)))
    if not order_match:
        diffs.append("列顺序不一致")

    if diffs:
        print(f"\n{'='*60}")
        print("列结构差异")
        print(f"{'='*60}")
        print("\n".join(diffs))

def _shape_str(v):
    if isinstance(v, np.ndarray):
        return str(v.shape)
    if isinstance(v, (list, tuple)):
        try:
            return f"list(len={len(v)})"
        except Exception:
            return "list"
    return "scalar"


def compare_structure(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    """对比两个 DataFrame 的数据结构（dtype + 样例值类型/形状）"""
    common_cols = set(df1.columns) & set(df2.columns)

    type_mismatches = []
    shape_mismatches = []
    details = []
    for col in sorted(common_cols):
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype

        v1 = df1[col].iloc[0] if len(df1) else None
        v2 = df2[col].iloc[0] if len(df2) else None
        s1 = _shape_str(v1)
        s2 = _shape_str(v2)

        if dtype1 != dtype2:
            type_mismatches.append((col, dtype1, dtype2))
        if s1 != s2:
            shape_mismatches.append((col, s1, s2))
        details.append(
            (
                col,
                str(dtype1),
                str(dtype2),
                type(v1).__name__ if v1 is not None else "None",
                type(v2).__name__ if v2 is not None else "None",
                s1,
                s2,
            )
        )

    if type_mismatches or shape_mismatches:
        print(f"\n{'='*60}")
        print("类型/形状差异")
        print(f"{'='*60}")

        if type_mismatches:
            print(f"数据类型不匹配的列 ({len(type_mismatches)}):")
            for col, t1, t2 in type_mismatches:
                print(f"  - {col}: {label1}={t1}, {label2}={t2}")

        if shape_mismatches:
            print(f"\n值形状不匹配的列 ({len(shape_mismatches)}，最多列出10):")
            for col, s1, s2 in shape_mismatches[:10]:
                print(f"  - {col}: {label1}={s1}, {label2}={s2}")


def compare_numeric_overlap(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    """对数值列做快速重叠检查（均值/最大/最小差异百分比）。"""
    numeric_cols = []
    for c in df1.columns:
        if c in df2.columns and pd.api.types.is_numeric_dtype(df1[c]) and pd.api.types.is_numeric_dtype(df2[c]):
            numeric_cols.append(c)

    diffs = []
    for c in numeric_cols:
        v1 = df1[c].dropna()
        v2 = df2[c].dropna()
        if len(v1) == 0 or len(v2) == 0:
            continue
        m1, m2 = v1.mean(), v2.mean()
        diff_pct = abs(m1 - m2) / (abs(m1) + 1e-9) * 100.0
        diffs.append((c, diff_pct, m1, m2))

    # 当前按需不输出数值差异
    return

def compare_content_sample(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    """对比前几行的内容结构"""
    print(f"\n{'='*60}")
    print("内容样本对比（前3行）")
    print(f"{'='*60}")
    
    # 对比基础列
    basic_cols = ['episode_index', 'tasks', 'length', 'data/chunk_index', 'data/file_index', 
                  'dataset_from_index', 'dataset_to_index']
    
    for col in basic_cols:
        if col in df1.columns and col in df2.columns:
            print(f"\n列: {col}")
            print(f"  {label1}: {df1[col].head(3).tolist()}")
            print(f"  {label2}: {df2[col].head(3).tolist()}")

def check_stats_columns(df: pd.DataFrame, label: str):
    """检查统计列的完整性"""
    stats_cols = [col for col in df.columns if col.startswith('stats/')]
    video_cols = [col for col in df.columns if col.startswith('videos/')]
    return len(stats_cols), len(video_cols)

# 主程序
if __name__ == "__main__":
    path1 = Path("C:/code/lerobot/dataset/lekiwi_box_20251203_202738/meta/episodes/chunk-000/file-000.parquet")
    path2 = Path("C:/code/lerobot/dataset/lekiwi_box_20251203_211441/meta/episodes/chunk-000/file-000.parquet")
    
    label1 = "参考文件 (202738)"
    label2 = "修复文件 (211441)"
    
    # 分析两个文件
    df1 = analyze_episodes_file(path1, label1)
    df2 = analyze_episodes_file(path2, label2)
    
    # 对比列结构（仅输出差异）
    compare_columns(df1, df2, label1, label2)

    # 对比数据类型 / 形状（仅输出差异）
    compare_structure(df1, df2, label1, label2)

    # 数值列重叠检查（仅输出差异）
    compare_numeric_overlap(df1, df2, label1, label2)

    # 统计/视频列数量差异（仅输出差异）
    stats1, video1 = check_stats_columns(df1, label1)
    stats2, video2 = check_stats_columns(df2, label2)
    stats_diff = stats1 != stats2 or video1 != video2
    if stats_diff:
        print(f"\n{'='*60}")
        print("统计/视频列数量差异")
        print(f"{'='*60}")
        print(f"  统计列: {label1}={stats1}, {label2}={stats2}")
        print(f"  视频列: {label1}={video1}, {label2}={video2}")

    # 最终总结（仅关键差异）
    cols_match = set(df1.columns) == set(df2.columns)
    order_match = list(df1.columns) == list(df2.columns)
    if (not cols_match) or (not order_match) or stats_diff or len(df1) != len(df2):
        print(f"\n{'='*60}")
        print("总结（存在差异）")
        print(f"{'='*60}")
        if not cols_match:
            print("  列集合不一致")
        if not order_match:
            print("  列顺序不一致")
        if stats_diff:
            print("  统计/视频列数量不一致")
        if len(df1) != len(df2):
            print(f"  Episodes 数量不同: {len(df1)} vs {len(df2)}")
    else:
        print(f"\n{'='*60}")
        print("总结：无差异（除非上方已输出差异）")
        print(f"{'='*60}")