'''
Description:
Author: 张轩誉
Date: 2024-07-10 13:53:51
LastEditors: 张轩誉
LastEditTime: 2024-07-10 14:46:13
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 获取当前文件的目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 设置输入文件和输出目录的路径
input_filename = os.path.join(current_file_dir, "result", "tracking_results.txt")
output_dir = os.path.join(current_file_dir, "Visualization", "tracking_picture")
table_filename = os.path.join(current_file_dir, "Visualization", "tracking_comparison.csv")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data = []
with open(input_filename, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        if len(parts) < 3:
            print(f"Skipping invalid line: {line.strip()}")
            continue
        
        entry = {}
        for part in parts:
            key, value = part.split(': ')
            entry[key.strip()] = value.strip()

        # 确保所需的键存在
        required_keys = ['Map ID', 'Algorithm', 'Controller', 'Average Tracking Error', 'Cumulative Error', 'Total Steps']
        if not all(key in entry for key in required_keys):
            print(f"Skipping line with missing data: {line.strip()}")
            continue

        # 尝试转换数值
        try:
            entry['Average Tracking Error'] = float(entry['Average Tracking Error'])
            entry['Cumulative Error'] = float(entry['Cumulative Error'])
            entry['Total Steps'] = int(entry['Total Steps'])
        except ValueError:
            print(f"Skipping line with invalid numerical data: {line.strip()}")
            continue

        data.append(entry)

# 转换为DataFrame
df = pd.DataFrame(data)

# 获取唯一的地图ID
map_ids = df['Map ID'].unique()

# 为每个地图ID创建一个图表
for map_id in map_ids:
    map_data = df[df['Map ID'] == map_id]

    # 获取唯一的算法和控制器
    algorithms = map_data['Algorithm'].unique()
    controllers = map_data['Controller'].unique()

    # 创建图表
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'Performance Comparison for Map ID: {map_id}')

    # 创建三个子图
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # 设置柱状图的位置
    x_pos = np.arange(len(controllers))
    y_pos = np.arange(len(algorithms))

    # 创建网格
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()

    # 设置柱状图的宽度
    width = depth = 0.4

    # 绘制三个指标的柱状图
    for ax, metric in zip([ax1, ax2, ax3], ['Average Tracking Error', 'Cumulative Error', 'Total Steps']):
        z_pos = []
        for alg in algorithms:
            for ctrl in controllers:
                value = map_data[(map_data['Algorithm'] == alg) & (map_data['Controller'] == ctrl)][metric].values
                z_pos.append(value[0] if len(value) > 0 else 0)

        ax.bar3d(x_pos, y_pos, np.zeros_like(z_pos), width, depth, z_pos, shade=True)
        ax.set_xticks(np.arange(len(controllers)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(controllers)
        ax.set_yticklabels(algorithms)
        ax.set_xlabel('Controller')
        ax.set_ylabel('Algorithm')
        ax.set_zlabel(metric)
        ax.set_title(f'{metric} Comparison')

    # 调整布局并保存图表
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f'tracking_comparison_map_{map_id}.png')
    plt.savefig(output_filename)
    plt.close()

# 保存汇总表格
df.to_csv(table_filename, index=False)

print("Visualization completed. Check the 'Visualization/tracking_picture' folder for images.")
print(f"The summary table has been saved to {table_filename}")
