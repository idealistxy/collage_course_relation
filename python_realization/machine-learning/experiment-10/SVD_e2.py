'''
Description:
Author: 张轩誉
Date: 2024-05-17 10:22:38
LastEditors: 张轩誉
LastEditTime: 2024-05-24 09:23:09
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def compress_image_svd(image_path, k):
    # 打开图像并转换为RGB格式
    image = Image.open(image_path).convert('RGB')
    image_matrix = np.array(image)

    # 初始化压缩后的图像矩阵
    compressed_image_matrix = np.zeros_like(image_matrix)

    # 对每个通道分别进行SVD压缩
    for channel in range(3):
        U, sigma, VT = np.linalg.svd(image_matrix[:, :, channel], full_matrices=False)
        U_k = U[:, :k]
        sigma_k = np.diag(sigma[:k])
        VT_k = VT[:k, :]
        compressed_channel = np.dot(U_k, np.dot(sigma_k, VT_k))
        # 确保压缩后的值在0-255范围内
        compressed_channel = np.clip(compressed_channel, 0, 255)
        compressed_image_matrix[:, :, channel] = compressed_channel

    return compressed_image_matrix.astype('uint8')


def plot_comparison(original_image_path, compressed_image_matrix):
    # 打开原始图像
    original_image = Image.open(original_image_path).convert('RGB')
    original_image_matrix = np.array(original_image)

    # 创建图像比较
    plt.figure(figsize=(10, 5))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image_matrix)
    plt.axis('off')

    # 显示压缩后的图像
    plt.subplot(1, 2, 2)
    plt.title('Compressed Image')
    plt.imshow(compressed_image_matrix)
    plt.axis('off')

    plt.show()


# 压缩图像并显示比较
image_path = 'ladybug.png'
k = 10  # 保留前k个奇异值

compressed_image_matrix = compress_image_svd(image_path, k)
plot_comparison(image_path, compressed_image_matrix)
