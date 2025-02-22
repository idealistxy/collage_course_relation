'''
Description: 
Author: 张轩誉
Date: 2024-03-15 11:08:01
LastEditors: 张轩誉
LastEditTime: 2024-03-15 14:32:49
'''
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN

warnings.filterwarnings('ignore')
np.random.seed(24)

# 随机生成6个点作为聚类中心
blob_centers = np.random.rand(6, 2) * 12 - 6

# 随机生成6个标准差
blob_std = np.random.rand(6) + 0.2

# 生成数据集
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std)
k = 6


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=2,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000,
                             show_centroids=True,
                             show_xlabels=True,
                             show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2", alpha=0.8)
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette_score = silhouette_score(X, kmeans_labels)


dbscan = DBSCAN(eps=0.5, min_samples=6)
dbscan.fit(X)
dbscan_labels = dbscan.fit_predict(X)
dbscan_silhouette_score = silhouette_score(X, dbscan_labels)

# 绘制聚类结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
plt.title('KMeans Clustering\nSilhouette ARG_Score: {:.2f}'.format(kmeans_silhouette_score))
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plot_decision_boundaries(kmeans, X)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering\nSilhouette ARG_Score: {:.2f}'.format(dbscan_silhouette_score))
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plot_decision_boundaries(kmeans, X)

plt.tight_layout()
plt.show()
