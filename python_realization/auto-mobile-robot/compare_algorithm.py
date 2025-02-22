'''
Description:
Author: 张轩誉
Date: 2024-07-10 13:19:41
LastEditors: 张轩誉
LastEditTime: 2024-07-10 14:00:30
'''
import matplotlib.pyplot as plt
import pandas as pd
import os

# 获取当前文件的目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 设置输入文件和输出目录的路径
input_filename = os.path.join(current_file_dir, "result", "planning_results.txt")
output_dir = os.path.join(current_file_dir, "Visualization", "planning_picture")
table_filename = os.path.join(current_file_dir, "Visualization", "planning_comparison.csv")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data = []
with open(input_filename, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        map_id = parts[0].split(': ')[1]
        algorithm = parts[1].split(': ')[1]
        path_quality = float(parts[2].split(': ')[1])
        planning_efficiency = float(parts[3].split(': ')[1].split(' ')[0])
        data.append([map_id, algorithm, path_quality, planning_efficiency])

# 转换为DataFrame
df = pd.DataFrame(data, columns=['Map ID', 'Algorithm', 'Path Quality', 'Planning Efficiency'])

# 获取唯一的地图ID
map_ids = df['Map ID'].unique()

# 创建一个列表来存储所有表格数据
all_tables = []

def set_ylim_safe(ax, data, buffer=0.1):
    max_val = max(data)
    if max_val == 0:
        ax.set_ylim(0, 1)  # 设置一个默认范围
    else:
        ax.set_ylim(0, max_val * (1 + buffer))

# 为每个地图ID创建一个图表和表格
for map_id in map_ids:
    map_data = df[df['Map ID'] == map_id]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Performance Comparison for Map ID: {map_id}')

    # 路径质量比较
    ax1.bar(map_data['Algorithm'], map_data['Path Quality'])
    ax1.set_title('Path Quality Comparison')
    ax1.set_ylabel('Path Quality')
    set_ylim_safe(ax1, map_data['Path Quality'], buffer=0.2)

    # 规划效率比较
    ax2.bar(map_data['Algorithm'], map_data['Planning Efficiency'])
    ax2.set_title('Planning Efficiency Comparison')
    ax2.set_ylabel('Planning Efficiency (seconds)')
    set_ylim_safe(ax2, map_data['Planning Efficiency'], buffer=0.2)

    # 在柱状图上添加数值标签
    for ax in [ax1, ax2]:
        for i, v in enumerate(ax.containers):
            ax.bar_label(v, fmt='%.2f', padding=3)

    # 调整布局并保存图表
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f'comparison_map_{map_id}.png')
    plt.savefig(output_filename)
    plt.close()

    # 创建表格数据并添加到列表中
    table_data = map_data[['Map ID', 'Algorithm', 'Path Quality', 'Planning Efficiency']]
    all_tables.append(table_data)

# 将所有表格数据合并为一个DataFrame
all_tables_df = pd.concat(all_tables, ignore_index=True)

# 保存为CSV，包含表头
all_tables_df.to_csv(table_filename, index=False)

print("Visualization completed. Check the 'result/planning_picture' folder for images.")
print(f"All tables have been saved to {table_filename}")
