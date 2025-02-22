'''
Description: Movies Classification
Author: 张轩誉
Date: 2024-03-01 09:54:53
LastEditors: 张轩誉
LastEditTime: 2024-03-01 11:33:32
'''
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

# 读取CSV文件
df = pd.read_csv('IMDB-Movie-Data.csv')

# 记录全体类别
all_class = []

# 获取唯一的类别
for genre in df['Genre']:
    all_class.extend(genre.split(','))
unique_class = set(all_class)

# 初始化一个字典来存储每个类别的计数,并计数。
genre_counts = {genre: all_class.count(genre) for genre in unique_class}

# 将字典转换为DataFrame
genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
genre_df = genre_df.sort_values(by='Count')

# 可视化
plt.bar(genre_df['Genre'], genre_df['Count'], color='orange')
plt.xlabel('Genre')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.title('Genre Distribution')
for i, value in enumerate(genre_df['Count']):
    plt.text(i, value + 0.5, str(value), horizontalalignment="center")
plt.show()
