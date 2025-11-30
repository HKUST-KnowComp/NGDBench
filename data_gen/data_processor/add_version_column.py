import pandas as pd

# 读取 kg.csv 文件
kg_path = "../gnd_dataset/PrimeKG/kg.csv"
print("正在读取 kg.csv 文件...")
df = pd.read_csv(kg_path)

print(f"原始数据形状: {df.shape}")
print(f"原始列名: {df.columns.tolist()}")

# 添加 version 列，所有值设置为 v1.0
df['version'] = 'v1.0'

print(f"添加 version 列后的形状: {df.shape}")
print(f"新列名: {df.columns.tolist()}")

# 保存回原文件
print("正在保存文件...")
df.to_csv(kg_path, index=False)

print("✓ 完成！已在 kg.csv 文件末尾添加 version 列，所有值设置为 v1.0")

