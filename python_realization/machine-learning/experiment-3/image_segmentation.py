'''
Description:
Author: 张轩誉
Date: 2024-03-15 10:03:29
LastEditors: 张轩誉
LastEditTime: 2024-03-15 14:46:13
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from skimage import img_as_ubyte

# 加载图像
img = load_sample_image("china.jpg")

# 转换图像数据类型并将像素值缩放到0-1之间
img = img_as_ubyte(img)
img = np.array(img, dtype=np.float64) / 255

# 将图像展平
w, h, d = original_shape = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))

# 设置不同的簇数即压缩后颜色数
n_clusters_list = [32, 16, 12, 10, 8, 6, 4, 2]

plt.figure(figsize=(16, 10))

# 循环处理图像
for k in range(len(n_clusters_list)+1):
    # 初始图象
    if k == 0:
        plt.subplot(3, 3, k+1)
        plt.imshow(img)
        plt.title('original image')
        plt.axis('off')
    else:
        n_clusters = n_clusters_list[k-1]
        # 对图像进行 KMeans 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(image_array)

        # 获取聚类中心和标签
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # 创建压缩后的图像
        compressed_image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                compressed_image[i][j] = centers[labels[label_idx]]
                label_idx += 1
        # 绘制压缩后的图像
        plt.subplot(3, 3, k+1)
        plt.imshow(compressed_image)
        plt.title('Compressed Image ({} Colors)'.format(n_clusters))
        plt.axis('off')

# 调整布局并显示图像
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()
