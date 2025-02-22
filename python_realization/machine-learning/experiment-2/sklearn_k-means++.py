'''
Description: k-means++
Author: 张轩誉
Date: 2024-03-08 10:59:57
LastEditors: 张轩誉
LastEditTime: 2024-03-08 13:37:05
'''
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import KMeans

# 读取文件数据
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
df = pd.read_excel('city_577513d4503f5adaea71.xls', skiprows=1)
last_to_columns = df.iloc[:3178, -2:]
data = last_to_columns.values


def plot_clusters_sklearn(data, labels, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids')
    plt.title('K-Means++ Clustering for city')
    plt.xlabel('X-longitude')
    plt.ylabel('Y-latitude')
    plt.legend()
    plt.show()


def kmeans_sklearn(S, k, max_iterations=100):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=max_iterations, random_state=42)
    labels = kmeans.fit_predict(S)
    centroids = kmeans.cluster_centers_

    return labels, centroids


def main_sklearn():
    num_clusters = 34

    # 使用 scikit-learn 进行 K-means++ 聚类
    labels, centroids = kmeans_sklearn(data, num_clusters)

    # 绘制聚类结果
    plot_clusters_sklearn(data, labels, centroids)


if __name__ == "__main__":
    main_sklearn()
