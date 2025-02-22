'''
Description: 
Author: 张轩誉
Date: 2023-12-26 18:18:43
LastEditors: 张轩誉
LastEditTime: 2024-01-15 20:11:59
'''
# flake8:noqa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.cm import ScalarMappable

# 读取CSV文件
df = pd.read_csv("D:/大三课程/计算机图形学/HW3/HW3/cars.csv") 

# 选择仅包含数字的列
numeric_columns = df.select_dtypes(include=['number'])

# 创建画布和子图
fig, ax = plt.subplots(figsize=(12, 8))


# 生成竖直坐标轴
for i, column in enumerate(numeric_columns.columns):
    y_ticks_positions = numeric_columns[column].values
    y_ticks_positions_min=np.min(y_ticks_positions[y_ticks_positions > 0])
    y_ticks_positions_max = np.nanmax(y_ticks_positions)
    y_ticks_positions_normalized = (y_ticks_positions - y_ticks_positions_min) / (
            y_ticks_positions_max - y_ticks_positions_min)
    # 绘制竖直坐标轴
    ax.plot([i, i], [0, 1], color='gray', linestyle='-', linewidth=2)

    # 在坐标轴上显示标签
    ax.text(i, 1.02, column, ha='center', va='bottom', rotation=0, fontsize=10, color='black')

    # 在坐标轴边上打印均匀分布的10个刻度
    ticks_indices = list(np.arange(0, 1.1,0.1))
    for index in ticks_indices:
        ax.text(i - 0.1, index, f'{index*(y_ticks_positions_max - y_ticks_positions_min)+y_ticks_positions_min:.2f}', ha='right', va='center', fontsize=8, color='black')

# 数据标准化
df_normalized = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())
# 获取颜色映射
cmap = plt.get_cmap('RdYlGn_r')
norm = Normalize(vmin=0, vmax=1)

# 遍历数据框的每一行，绘制平行线
for i, (idx, row) in enumerate(df_normalized.iterrows()):
    color_val =row.iloc[0]  # 使用第一列的归一化数值作为颜色映射的输入
    points = list(enumerate(row))
    for j, point in points[:-1]:
        ax.plot([j, j + 1], [point, points[j + 1][1]], color=cmap(color_val), alpha=0.7, linewidth=0.5)
"""
# 遍历数据框的每一行，绘制平行线
for i, (idx, row) in enumerate(df_normalized.iterrows()):
    points = list(enumerate(row))
    color_val = norm(i)
    for j, point in points[:-1]:
        ax.plot([j, j + 1], [point, points[j + 1][1]], color=cmap(color_val), alpha=0.7, linewidth=0.5)
"""
# 设置x轴刻度和标签
ax.set_xticks(range(len(numeric_columns.columns)))
ax.set_xticklabels([])  # 隐藏x轴刻度标签

# 添加坐标轴标签
ax.set_xlabel("attribute")

# 添加标题
plt.title('Visualization of the parallel coordinate axes', fontsize=16)

# 添加颜色对照标准
sm = ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Color Legend')
# 手动调整布局
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 显示图形
plt.show()
