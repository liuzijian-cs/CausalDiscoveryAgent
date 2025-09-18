import pandas as pd
import re

# 1) 读取原始 .data 文件，注意 ? 表示缺失值
df = pd.read_csv(
    "arrhythmia.data",
    header=None,
    sep=",",
    na_values="?",   # 关键：将 ? 转为 NaN
    engine="python"
)

# 2) 校验列数
if df.shape[1] != 280:
    raise ValueError(f"列数异常，读取到 {df.shape[1]} 列，期望 280 列")

# 3) 尝试从 names 文件解析列名（可选）
def parse_names_to_headers(names_path):
    try:
        text = open(names_path, encoding="utf-8").read()
    except Exception:
        return None

    pat = re.compile(r"^\s*(\d{1,3})\s+([^:]+):", re.M)
    names = {}
    for m in pat.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= 279:
            names[idx] = re.sub(r"\s+", "_", m.group(2).strip())

    if len(names) < 200:
        return None  # 解析列名不足，放弃
    headers = [names.get(i, f"f{i:03d}") for i in range(1, 280)]
    return headers

headers = parse_names_to_headers("arrhythmia.names")
if headers is None:
    headers = [f"f{i:03d}" for i in range(1, 280)]
headers.append("class")
df.columns = headers

# 4) 缺失值处理：这里示范用中位数填充连续变量，类别变量用众数填充
# 如果不想填充可直接跳过，让后续模型自行处理
for col in df.columns[:-1]:  # 最后一列是标签
    if df[col].dtype.kind in "biufc":  # 数值列
        df[col] = df[col].fillna(df[col].median())
    else:  # 类别列
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# 5) 导出为标准 CSV
df.to_csv("arrhythmia_clean.csv", index=False)
print("✅ 已保存 arrhythmia_clean.csv")
print(df.info())
print(df.head())