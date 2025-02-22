import numpy as np
import matplotlib.pyplot as plt


# 栅格区域大小
shape = (64, 64)


# 随机生成5个三角形
def generate_random_triangle():
    while True:
        # 在栅格范围内随机生成三个点
        p1 = np.random.randint(0, shape[0], size=2)
        p2 = np.random.randint(0, shape[1], size=2)
        p3 = np.random.randint(0, shape[0], size=2)
        # 确保这三个点不共线
        if (p2[1] - p1[1]) * (p3[0] - p1[0]
                              ) != (p3[1] - p1[1]) * (p2[0] - p1[0]):
            if (p2[1] - p1[1]) * (p3[0] - p1[0]
                                  ) != -1*(p3[1] - p1[1]) * (p2[0] - p1[0]):
                plt.plot(p1[0], p1[1], 'ro')
                plt.plot(p2[0], p2[1], 'ro')
                plt.plot(p3[0], p3[1], 'ro')
            break
    return p1, p2, p3


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def is_inside_triangle(x1, y1, x2, y2, x3, y3, x, y):
    total_area = area(x1, y1, x2, y2, x3, y3)
    area1 = area(x, y, x2, y2, x3, y3)
    area2 = area(x1, y1, x, y, x3, y3)
    area3 = area(x1, y1, x2, y2, x, y)
    return total_area == (area1 + area2 + area3)


# 计算一个点是否在三角形内
def point_in_triangle(point, triangle):
    p1, p2, p3 = triangle
    return is_inside_triangle(p1[0], p1[1],
                              p2[0], p2[1], p3[0], p3[1], point[0], point[1])


def takef(point):
    return point[0]


def takes(point):
    return point[1]


def sampling(grid, point, triangle):
    brightness = 0
    if 0 <= point[0] < len(grid) and 0 <= point[1] < len(grid[0]):
        mat = [(-3/8, -3/8), (-1/8, -3/8),
               (1/8, -3/8), (3/8, -3/8),
               (-3/8, -1/8), (-1/8, -1/8),
               (1/8, -1/8), (3/8, -1/8),
               (-3/8, 1/8), (-1/8, 1/8),
               (1/8, 1/8), (3/8, 1/8),
               (-3/8, 3/8), (-1/8, 3/8),
               (1/8, 3/8), (3/8, 3/8)]
        brightness = 0
        for dx, dy in mat:
            nx2, ny2 = point[0] + dx, point[1] + dy
            if point_in_triangle([nx2, ny2], triangle):
                # plt.plot(nx2, ny2, 'go') # 去掉注释显示采样点
                brightness += 1/16
        grid[point[1], point[0]] = brightness
    return brightness


def midcheck(grid, triangle):
    p1, p2, p3 = triangle
    listlr = [p1, p2, p3]
    listud = [p1, p2, p3]
    listud.sort(key=takes)
    listlr.sort(key=takef)
    L = listlr[0][0]
    R = listlr[2][0]
    U = listud[0][1]
    D = listud[2][1]
    SX = listlr[1][0]
    SY = listlr[1][1]
    lhb = rhb = SX
    for j in range(SY, U-1, -1):
        flag = 0
        for i in range(SX, L-1, -1):
            if sampling(grid, [i, j], triangle) == 0:
                if i == SX:
                    flag = 1
                if flag == 0:
                    break
                if flag == 1 and lhb > L:
                    continue
            else:
                if lhb > i:
                    lhb = i            # 记录所到达过的最左侧
        flag = 0
        for i in range(SX+1, R+1, 1):
            if sampling(grid, [i, j], triangle) == 0:
                if i == SX+1:
                    flag = 1
                if flag == 0:
                    break
                if flag == 1 and rhb < R:
                    continue
            else:
                if rhb < i:
                    rhb = i            # 记录所到达过的最右侧
    for j in range(SY+1, D+1, 1):
        flag = 0
        for i in range(SX, L-1, -1):
            if sampling(grid, [i, j], triangle) == 0:
                if i == SX:
                    flag = 1
                if flag == 0:
                    break
                if flag == 1 and lhb > L:
                    continue
            else:
                if lhb > i:
                    lhb = i            # 记录所到达过的最左侧
        flag = 0
        for i in range(SX+1, R+1, 1):
            if sampling(grid, [i, j], triangle) == 0:
                if i == SX+1:
                    flag = 1
                if flag == 0:
                    break
                if flag == 1 and rhb < R:
                    continue
            else:
                if rhb < i:
                    rhb = i            # 记录所到达过的最右侧


# 检查每个栅格点是否在任何三角形内
def check_grid_points():
    grid = np.zeros(shape, dtype=float)
    centroids = []
    for _ in range(1):
        triangle = generate_random_triangle()
        p1, p2, p3 = triangle
        print(p1, p2, p3)
        centroids = np.mean(triangle, axis=0)
        plt.plot(int(centroids[0]), int(centroids[1]), 'bo')
        midcheck(grid, triangle)
    return grid, centroids


# 可视化结果
for i in range(5):
    grid, centroids = check_grid_points()
    plt.imshow(grid, cmap='gray')
    plt.show()
