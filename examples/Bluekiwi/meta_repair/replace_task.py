import pandas as pd
from pathlib import Path

dataset_root = "dataset/lekiwi_fixed_newmerged"
tasks_path = Path(dataset_root) / "meta/tasks.parquet"

# 读取任务文件
df = pd.read_parquet(tasks_path)
print("修改前:")
print(df)
print()

# 统一的新提示词
new_task = "Pick up the object and put it into the box."

# 修改 task 列
df["task"] = new_task

# 重置索引并设置新的索引为 task 列的值
df = df.reset_index(drop=True)
df.index = [new_task] * len(df)
df.index.name = "task"

print("修改后:")
print(df)

# 保存
df.to_parquet(tasks_path)
print(f"\n✓ 已更新: {tasks_path}")