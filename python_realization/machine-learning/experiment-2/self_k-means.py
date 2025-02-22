'''
Description: k-means
Author: 张轩誉
Date: 2024-03-08 10:59:57
LastEditors: 张轩誉
LastEditTime: 2024-03-08 13:57:25
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 读取文件数据
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
df = pd.read_excel('city_577513d4503f5adaea71.xls', skiprows=1)
last_to_columns = df.iloc[:3178, -2:]
data = last_to_columns.values


# 计算数据点与中心距离
def calculate_distance_matrix(S, centroids):
    distances = np.zeros((S.shape[0], centroids.shape[0]))
    for i in range(S.shape[0]):
        for j in range(centroids.shape[0]):
            distances[i, j] = L2(S[i], centroids[j])
    return distances


# 计算中心距离的平均偏移量
def calculate_centroid_change(old_centroids, centroids):
    total_distance = 0
    for i in range(old_centroids.shape[0]):
        total_distance += L2(old_centroids[i], centroids[i])
    return total_distance / old_centroids.shape[0]


# 欧氏距离
def L2(vecXi, vecXj):
    a, b = vecXi
    c, d = vecXj
    return pow((pow(a - c, 2) + pow(b - d, 2)), 0.5)


def plot_clusters(data, labels, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids')
    plt.title('K-Means Clustering for city')
    plt.xlabel('X-longitude')
    plt.ylabel('Y-latitude')
    plt.legend()
    plt.show()


# Kmeans实现
def k_means_clustering(S, k, max_iterations=100):
    num_points, _ = S.shape

    # 随机初始化 k 个中心点
    centroids = S[np.random.choice(num_points, k, replace=False)]

    sse_list = []  # 储存迭代的SSE

    for iteration in range(max_iterations):
        # 计算每个点到各个中心点的距离
        distances = calculate_distance_matrix(S, centroids)

        # 分配每个点到最近的中心点
        clusterCents = np.argmin(distances, axis=1)

        # 保存旧的中心点位置
        old_centroids = centroids.copy()

        # 更新中心点为每个簇的平均值
        for i in range(k):
            centroids[i] = np.mean(S[clusterCents == i], axis=0)

        # 计算SSE
        sse = np.sum((S - centroids[clusterCents]) ** 2)
        sse_list.append(sse)

        # 计算中心点的变化
        centroid_change = calculate_centroid_change(old_centroids, centroids)

        # 打印当前迭代的中心点变化和SSE
        print(f"Iteration {iteration + 1}, Centroid Change: {centroid_change}, SSE: {sse}")

        # 当中心点的不变化时停止迭代
        if centroid_change == 0:
            break

    return clusterCents, sse_list


def main():
    num_clusters = 34

    # 使用 k-means 进行聚类
    labels, sse_list = k_means_clustering(data, num_clusters)

    # 获取聚类中心点
    centroids = np.array([np.mean(data[labels == i], axis=0) for i in range(num_clusters)])

    # 绘制聚类结果
    plot_clusters(data, labels, centroids)

    # 打印最终SSE
    final_sse = sse_list[-1]
    print(f"Final SSE: {final_sse}")


if __name__ == "__main__":
    main()
