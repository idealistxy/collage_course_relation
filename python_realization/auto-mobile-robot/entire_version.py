import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from queue import Queue, PriorityQueue
import random
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import splprep, splev
import time
import os
import hashlib

# 定义网格大小
grid_size = (30, 30)
grid = np.zeros(grid_size)

# 用于存储起点、终点和障碍物
start = None
goal = None
obstacles = []
drawing = False
x0, y0 = None, None
history = []

# 绘制初始网格
fig, ax = plt.subplots(figsize=(30, 30))
ax.imshow(grid.T, origin='lower', cmap='gray')
ax.grid(True)
plt.xticks(np.arange(0, grid_size[0], 1))
plt.yticks(np.arange(0, grid_size[1], 1))
plt.title('Click to set start, end, obstacle')

# 定义A*8个方向
directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# 确保起点和终点之间有路径
def is_reachable(grid_size, start, goal, obstacles):
    visited = set()
    q = Queue()
    q.put(start)
    visited.add(start)

    while not q.empty():
        current = q.get()
        if current == goal:
            return True
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    q.put(neighbor)
    return False


# 路径规划和路径跟踪部分

# 曼哈顿距离启发式
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# 定义二轮差分小车类
class DifferentialDriveRobot:
    def __init__(self, x, y, theta, width=0.8, length=0.8, wheel_radius=0.2, max_speed=1.0, max_angular_speed=np.pi/2):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.length = length
        self.wheel_radius = wheel_radius
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed

    def update_pose(self, v, omega, dt):
        """
        更新机器人姿态
        v: 线速度
        omega: 角速度
        dt: 时间步长
        """
        # 限制速度在允许范围内
        v = np.clip(v, -self.max_speed, self.max_speed)
        omega = np.clip(omega, -self.max_angular_speed, self.max_angular_speed)

        # 更新位置和方向
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # 将角度归一化到 [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def inverse_kinematics(self, v, omega):
        """
        逆运动学：从线速度和角速度计算左右轮速度
        """
        vl = v - (self.width / 2) * omega
        vr = v + (self.width / 2) * omega
        return vl, vr

    def forward_kinematics(self, vl, vr):
        """
        正运动学：从左右轮速度计算线速度和角速度
        """
        v = (vr + vl) / 2
        omega = (vr - vl) / self.width
        return v, omega

    def get_outline(self):
        corners = [
            (-self.length/2, -self.width/2),
            (self.length/2, -self.width/2),
            (self.length/2, self.width/2),
            (-self.length/2, self.width/2)
        ]
        rotated_corners = [self.rotate_point(p) for p in corners]
        translated_corners = [(p[0] + self.x, p[1] + self.y) for p in rotated_corners]
        return translated_corners

    def rotate_point(self, point):
        x, y = point
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        return (x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta)

    def get_wheels(self):
        wheel_width = 0.2
        wheel_radius = 0.2
        left_wheel_center = self.rotate_point((-self.length/4, -self.width/2))
        right_wheel_center = self.rotate_point((-self.length/4, self.width/2))
        left_wheel = (left_wheel_center[0] + self.x, left_wheel_center[1] + self.y)
        right_wheel = (right_wheel_center[0] + self.x, right_wheel_center[1] + self.y)
        return left_wheel, right_wheel, wheel_width, wheel_radius

    def move_to_pose(self, target_x, target_y, target_theta):
        """
        移动到目标姿态的控制逻辑
        """
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)

        angle_to_goal = np.arctan2(dy, dx)
        angle_error = self.normalize_angle(angle_to_goal - self.theta)

        # 简单的比例控制
        k_v = 0.5  # 线速度增益
        k_omega = 1.0  # 角速度增益

        v = k_v * distance
        omega = k_omega * angle_error

        return v, omega

    @staticmethod
    def normalize_angle(angle):
        """
        将角度归一化到 [-pi, pi]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))


# 扩展障碍(考虑自主移动机器人形态)
def inflate_obstacles(obstacles, robot_size):
    inflated_obstacles = set(obstacles)
    for ox, oy in obstacles:
        for dx in range(-robot_size, robot_size + 1):
            for dy in range(-robot_size, robot_size + 1):
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                    inflated_obstacles.add((nx, ny))
    return inflated_obstacles


# A*搜索
def a_star_search(start, goal, grid, obstacles, robot_size):
    inflated_obstacles = inflate_obstacles(obstacles, robot_size)
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            break

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]:
                if neighbor in inflated_obstacles:
                    continue
                new_cost = cost_so_far[current] + (1 if (dx == 0 or dy == 0) else 1.414)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    open_set.put((priority, neighbor))
                    came_from[neighbor] = current

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


# 添加蚁群优化算法
def ant_colony_optimization(start, goal, obstacles, robot_size, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=1.0):
    obstacles = inflate_obstacles(obstacles, robot_size)

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def create_graph():
        graph = {}
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if (x, y) not in obstacles:
                    graph[(x, y)] = {}
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and (nx, ny) not in obstacles:
                            graph[(x, y)][(nx, ny)] = distance((x, y), (nx, ny))
        return graph

    graph = create_graph()
    pheromone = {(i, j): {(x, y): 1.0 for x, y in graph[i, j]} for i, j in graph}

    best_path = None
    best_path_length = float('inf')

    for _ in range(n_iterations):
        paths = []
        path_lengths = []

        for _ in range(n_ants):
            path = [start]
            path_length = 0
            current = start

            while current != goal:
                if current not in graph:
                    break
                candidates = list(graph[current].keys())
                if not candidates:
                    break

                probabilities = []
                for next_node in candidates:
                    tau = pheromone[current][next_node]
                    eta = 1.0 / graph[current][next_node]
                    probabilities.append((tau**alpha) * (eta**beta))

                probabilities = np.array(probabilities) / sum(probabilities)
                next_node = random.choices(candidates, weights=probabilities)[0]

                path.append(next_node)
                path_length += graph[current][next_node]
                current = next_node

            if current == goal:
                paths.append(path)
                path_lengths.append(path_length)

                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

        # 更新信息素
        for i, j in pheromone:
            for x, y in pheromone[i, j]:
                pheromone[i, j][x, y] *= (1.0 - evaporation_rate)

        for path, path_length in zip(paths, path_lengths):
            for i in range(len(path) - 1):
                current, next_node = path[i], path[i+1]
                pheromone[current][next_node] += Q / path_length

    return best_path if best_path else None


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def dist(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def get_random_node():
    return Node(random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))


def nearest_node(nodes, random_node):
    nearest = nodes[0]
    for node in nodes:
        if dist(node, random_node) < dist(nearest, random_node):
            nearest = node
    return nearest


def steer(from_node, to_node, max_distance=1.0):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    distance = np.sqrt(dx**2 + dy**2)
    if distance > max_distance:
        dx = dx * max_distance / distance
        dy = dy * max_distance / distance
    return Node(from_node.x + dx, from_node.y + dy)


def is_path_free(from_node, to_node, obstacles):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    distance = np.sqrt(dx**2 + dy**2)
    if distance == 0:
        return True  # 如果两个节点重合，认为路径是自由的
    steps = max(int(distance * 2), 1)  # 确保至少有一个步骤
    for i in range(steps + 1):
        t = i / steps
        x = from_node.x + dx * t
        y = from_node.y + dy * t
        if (int(round(x)), int(round(y))) in obstacles:
            return False
    return True


# 快速随机树算法
def rrt(start, goal, obstacles, robot_size, max_iter=10000):
    inflated_obstacles = inflate_obstacles(obstacles, robot_size)
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    nodes = [start_node]

    if (int(round(start[0])), int(round(start[1]))) in inflated_obstacles or (int(round(goal[0])), int(round(goal[1]))) in inflated_obstacles:
        print("Start or goal is in obstacle")
        return []

    for i in range(max_iter):
        rand_node = get_random_node()
        nearest = nearest_node(nodes, rand_node)
        new_node = steer(nearest, rand_node)

        if 0 <= new_node.x < grid_size[0] and 0 <= new_node.y < grid_size[1]:  # 确保新节点在网格内
            if (int(round(new_node.x)), int(round(new_node.y))) not in inflated_obstacles and is_path_free(nearest, new_node, inflated_obstacles):
                new_node.parent = nearest
                nodes.append(new_node)

                if dist(new_node, goal_node) < 0.5:  # 使用更小的阈值
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    break

    if i == max_iter - 1:
        print("RRT failed to find a path")
        return []

    path = []
    node = goal_node
    while node:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path


def increase_path_points(path, num_points=100):
    """通过线性插值增加路径点数量"""
    xs, ys = zip(*path)
    interp_func_x = np.interp(np.linspace(0, len(xs) - 1, num_points), range(len(xs)), xs)
    interp_func_y = np.interp(np.linspace(0, len(ys) - 1, num_points), range(len(ys)), ys)
    return list(zip(interp_func_x, interp_func_y))


def smooth_path(path):
    """对路径进行平滑处理"""
    if len(path) < 4:
        print("Path too short for Bezier interpolation, increasing points using linear interpolation.")
        path = increase_path_points(path)

    xs, ys = zip(*path)

    try:
        # 使用贝塞尔曲线进行插值
        tck, u = splprep([xs, ys], s=0)
        new_indices = np.linspace(0, 1, len(xs) * 10)
        new_xs, new_ys = splev(new_indices, tck)

    except ValueError as e:
        print(f"Error with Bezier interpolation: {e}. Using linear interpolation instead.")
        new_xs, new_ys = xs, ys

    smoothed_path = list(zip(new_xs, new_ys))
    return smoothed_path


def calculate_path_quality(path):
    """计算路径质量(长度)"""
    if not path:
        return 0
    path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
    return path_length


def calculate_planning_efficiency(start_time, end_time):
    """计算规划效率(时间)"""
    return end_time - start_time


def save_planning_results(map_id, algorithm, path_quality, planning_efficiency):
    """保存规划结果到文本文件"""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_file_dir, "result/planning_results.txt")
    try:
        with open(filename, "a") as f:
            f.write(f"Map ID: {map_id}, Algorithm: {algorithm}, Path Quality: {path_quality:.2f}, Planning Efficiency: {planning_efficiency:.4f} seconds\n")
            f.flush()
        print("成功写入文件")
    except IOError as e:
        print(f"写入文件时出错: {e}")


def save_tracking_results(map_id, algorithm, controller, average_tracking_error, cumulative_error, total_steps):
    """保存追踪结果到文本文件"""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_file_dir, "result/tracking_results.txt")
    try:
        with open(filename, "a") as f:
            f.write(f"Map ID: {map_id}, Controller: {controller}, "
                    f"Algorithm: {algorithm}, "
                    f"Average Tracking Error: {average_tracking_error:.4f}, "
                    f"Cumulative Error: {cumulative_error:.4f}, "
                    f"Total Steps: {total_steps}\n")
        print("成功写入文件")
    except IOError as e:
        print(f"写入文件时出错: {e}")


def generate_map_id(start, goal, obstacles):
    """根据起点、终点和障碍物生成唯一的地图ID"""
    map_config = str(start) + str(goal) + str(sorted(obstacles))
    return hashlib.md5(map_config.encode()).hexdigest()[:8]  # 使用前8个字符作为ID


# 比例积分控制器
class PIDController:
    def __init__(self, kp_linear, ki_linear, kd_linear, kp_angular, ki_angular, kd_angular, goal_tolerance=0.1, max_angular_speed=np.pi/2, max_linear_speed=1.0):
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular
        self.goal_tolerance = goal_tolerance
        self.max_angular_speed = max_angular_speed
        self.max_linear_speed = max_linear_speed
        self.prev_error_linear = 0
        self.prev_error_angular = 0
        self.integral_linear = 0
        self.integral_angular = 0

    def control(self, current, target):
        dx = target[0] - current[0]
        dy = target[1] - current[1]

        distance = np.sqrt(dx**2 + dy**2)
        desired_angle = np.arctan2(dy, dx)

        current_angle = np.arctan2(np.sin(current[2]), np.cos(current[2]))
        angle_error = self.normalize_angle(desired_angle - current_angle)

        # 线速度PID控制
        self.integral_linear += distance
        derivative_linear = distance - self.prev_error_linear
        v = self.kp_linear * distance + self.ki_linear * self.integral_linear + self.kd_linear * derivative_linear

        # 角速度PID控制
        self.integral_angular += angle_error
        derivative_angular = angle_error - self.prev_error_angular
        omega = self.kp_angular * angle_error + self.ki_angular * self.integral_angular + self.kd_angular * derivative_angular

        # 添加大角度旋转逻辑
        if abs(angle_error) > np.pi/4:  # 如果角度误差大于45度
            v = 0  # 停止平移
            omega = np.clip(self.kp_angular * angle_error, -self.max_angular_speed, self.max_angular_speed)
        else:
            v = np.clip(v, -self.max_linear_speed, self.max_linear_speed)
            omega = np.clip(omega, -self.max_angular_speed, self.max_angular_speed)

        self.prev_error_linear = distance
        self.prev_error_angular = angle_error

        return v, omega

    @staticmethod
    def normalize_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def is_goal_reached(self, current, target):
        distance = np.linalg.norm(np.array(target[:2]) - np.array(current[:2]))
        angle_diff = self.normalize_angle(target[2] - current[2])
        return distance < self.goal_tolerance and abs(angle_diff) < np.pi/18  # 10度


class PurePursuitController:
    def __init__(self, lookahead_distance, linear_velocity, goal_tolerance):
        self.lookahead_distance = lookahead_distance
        self.linear_velocity = linear_velocity
        self.goal_tolerance = goal_tolerance
        self.simulation_start_time = None
        self.last_target_idx = 0

    def control(self, path, robot_pose):

        # 检查是否到达终点
        if self.is_goal_reached(path[-1], robot_pose):
            return 0, 0, path[-1]  # 如果到达终点，停止移动

        # 找到最近的路径点
        closest_idx = self.find_closest_point(path, robot_pose)

        # 动态调整前视距离
        distance_to_goal = np.linalg.norm(np.array(path[-1][:2]) - robot_pose[:2])
        self.lookahead_distance = min(self.lookahead_distance, max(0.5, distance_to_goal))

        # 寻找目标点
        target_idx = self.find_target_point(path, robot_pose, closest_idx)
        target_point = path[target_idx]

        # 计算机器人局部坐标系中的目标点位置
        dx = target_point[0] - robot_pose[0]
        dy = target_point[1] - robot_pose[1]
        target_y = -np.sin(robot_pose[2]) * dx + np.cos(robot_pose[2]) * dy

        # 计算横向误差
        lateral_error = target_y

        # 计算前视距离
        actual_lookahead = np.sqrt(dx**2 + dy**2)

        # 计算曲率
        curvature = 2 * lateral_error / (actual_lookahead**2)

        # 计算角速度
        omega = curvature * self.linear_velocity

        # 动态调整线速度
        adjusted_velocity = self.adjust_velocity(curvature, path, robot_pose)

        return adjusted_velocity, omega, target_point

    def find_closest_point(self, path, robot_pose):
        distances = [np.linalg.norm(np.array(p[:2]) - robot_pose[:2]) for p in path]
        return np.argmin(distances)

    def find_target_point(self, path, robot_pose, start_idx):
        for i in range(start_idx, len(path)):
            if np.linalg.norm(np.array(path[i][:2]) - robot_pose[:2]) > self.lookahead_distance:
                return i
        return len(path) - 1

    def adjust_velocity(self, curvature, path, robot_pose):
        # 根据曲率调整速度，转弯时降低速度
        max_velocity = self.linear_velocity * 5
        min_velocity = 0.01 * self.linear_velocity
        adjusted_velocity = max_velocity / (1 + abs(curvature) * 5)  # 5 是一个可调整的参数

        # 根据距离目标的远近进一步调整速度
        distance_to_goal = np.linalg.norm(np.array(path[-1][:2]) - robot_pose[:2])
        velocity_factor = min(1.0, distance_to_goal / self.lookahead_distance)
        adjusted_velocity *= velocity_factor

        return np.clip(adjusted_velocity, min_velocity, max_velocity)

    def is_goal_reached(self, goal, robot_pose):
        distance_to_goal = np.linalg.norm(np.array(goal[:2]) - robot_pose[:2])
        angle_to_goal = np.arctan2(goal[1] - robot_pose[1], goal[0] - robot_pose[0])
        angle_diff = self.normalize_angle(angle_to_goal - robot_pose[2])
        return distance_to_goal < self.goal_tolerance and abs(angle_diff) < np.pi/18  # 10度

    @staticmethod
    def normalize_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))


# 界面交互和主程序

# 定义路径规划算法选项
algorithm_options = ['A* Search', 'ACO', 'RRT']
controller_options = ['PID', 'Pure Pursuit']


# 创建选择框
algorithm_radio = RadioButtons(plt.axes([0.75, 0.25, 0.15, 0.15]), algorithm_options)
controller_radio = RadioButtons(plt.axes([0.75, 0.05, 0.15, 0.15]), controller_options)


def onclick(event):
    global start, goal, obstacles, drawing, x0, y0, history
    ix, iy = int(event.xdata), int(event.ydata)

    if event.button == 1:
        if start is None:   # 左键单击绘制起点
            start = (ix, iy)
            ax.scatter(ix, iy, color='green', s=100, label='Start')
            plt.draw()
            history.append(('start', start))
        elif goal is None:  # 左键再次单击绘制终点
            goal = (ix, iy)
            ax.scatter(ix, iy, color='red', s=100, label='Goal')
            plt.draw()
            history.append(('goal', goal))
    elif event.button == 3:  # 右键单击选择障碍端点
        if not drawing:  # 如果当前没有正在绘制障碍，则记录起点
            x0, y0 = ix, iy
            drawing = True
        else:  # 如果当前已经有起点，则记录终点并绘制障碍
            x1, y1 = ix, iy
            drawing = False

            # 使用 Bresenham's 线算法绘制两点间的障碍
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            line_obstacles = []
            while True:
                if (x0, y0) not in obstacles and (x0, y0) != start and (x0, y0) != goal:
                    obstacles.append((x0, y0))
                    ax.scatter(x0, y0, color='white', s=100)
                    plt.draw()
                    line_obstacles.append((x0, y0))
                if (x0, y0) == (x1, y1):
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            history.append(('obstacle', line_obstacles))


# 定义一个全局变量来存储当前路径
current_paths = []


def confirm(event):
    global current_paths, map_id

    if not start or not goal:
        print("Please select start and goal points.")
        return

    if not is_reachable(grid_size, start, goal, obstacles):
        print("No path exists between the start and goal with the current obstacles.")
        return

    map_id = generate_map_id(start, goal, obstacles)

    algorithm = algorithm_radio.value_selected
    smooth = smooth_button.get_status()[0]  # 是否启用平滑功能

    # 清除现有路径
    current_paths.clear()

    start_time = time.time()

    if algorithm == 'A* Search':
        path = a_star_search(start, goal, grid, obstacles, robot_size=1)
        path_color = 'b-'  # A* Search 路径颜色为蓝色
    elif algorithm == 'ACO':
        path = ant_colony_optimization(start, goal, obstacles, robot_size=1)
        path_color = 'y-'   # ACO 路径颜色为黄色
    elif algorithm == 'RRT':
        path = rrt(start, goal, obstacles, robot_size=1)
        path_color = 'm-'  # RRT 路径颜色为紫色
    else:
        print("Invalid algorithm selected.")
        return

    end_time = time.time()

    if path:
        if smooth:
            path = smooth_path(path)  # 对路径进行平滑处理

        path_quality = calculate_path_quality(path)
        planning_efficiency = calculate_planning_efficiency(start_time, end_time)

        save_planning_results(map_id, algorithm, path_quality, planning_efficiency)

        # 存储当前路径
        current_paths.append((path, algorithm, path_color))

        # 绘制路径
        xs, ys = zip(*path)
        ax.plot(xs, ys, path_color, alpha=0.7, label=f'{algorithm}')  # 绘制路径

    plt.legend()  # 添加图例
    fig.canvas.draw()


# reset 函数保持不变
def reset(event):
    global start, goal, obstacles, map_id
    start, goal, obstacles = None, None, []
    map_id = None
    ax.clear()
    ax.imshow(grid.T, origin='lower', cmap='gray')
    ax.grid(True)
    ax.set_xticks(np.arange(0, grid_size[0], 1))
    ax.set_yticks(np.arange(0, grid_size[1], 1))
    ax.set_title('Click to set start, end, obstacle')

    # 清除路径、起终点、障碍物等
    for obstacle in obstacles:
        ax.scatter(obstacle[0], obstacle[1], color='gray', s=100)
    if start:
        ax.scatter(start[0], start[1], color='green', s=100, label='Start')
    if goal:
        ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')
    fig.canvas.draw()


# 修改模拟路径函数
def simulate_path(path):
    fig.canvas.flush_events()

    # 创建二轮差分小车实例
    robot = DifferentialDriveRobot(path[0][0], path[0][1], 0)

    # 创建小车主体和轮子的图形对象
    robot_body = Polygon(robot.get_outline(), fill=False, edgecolor='r', linewidth=2)
    left_wheel, right_wheel, wheel_width, wheel_radius = robot.get_wheels()
    left_wheel = Circle(left_wheel, wheel_radius, fill=True, color='black')
    right_wheel = Circle(right_wheel, wheel_radius, fill=True, color='black')

    ax.add_patch(robot_body)
    ax.add_patch(left_wheel)
    ax.add_patch(right_wheel)

    for i in range(len(path) - 1):
        current_point = path[i]
        next_point = path[i + 1]

        # 计算方向角
        dx = next_point[0] - current_point[0]
        dy = next_point[1] - current_point[1]
        theta = np.arctan2(dy, dx)

        # 更新小车位置和方向
        robot.update_pose(current_point[0], current_point[1], theta)

        # 更新小车主体
        robot_body.set_xy(robot.get_outline())

        # 更新轮子位置
        left_wheel_pos, right_wheel_pos, _, _ = robot.get_wheels()
        left_wheel.center = left_wheel_pos
        right_wheel.center = right_wheel_pos

        fig.canvas.draw_idle()
        plt.pause(0.1)  # 调整暂停时间以控制模拟速度

    # 更新小车到最终位置
    robot.update_pose(path[-1][0], path[-1][1], theta)
    robot_body.set_xy(robot.get_outline())
    left_wheel_pos, right_wheel_pos, _, _ = robot.get_wheels()
    left_wheel.center = left_wheel_pos
    right_wheel.center = right_wheel_pos

    fig.canvas.draw_idle()
    plt.pause(0.5)

    # 清除小车图形
    robot_body.remove()
    left_wheel.remove()
    right_wheel.remove()


def simulate(event):
    global map_id
    if not current_paths:
        print("No path to simulate. Please generate a path first.")
        return

    path, algorithm, _ = current_paths[-1]
    selected_controller = controller_radio.value_selected

    robot = DifferentialDriveRobot(path[0][0], path[0][1], 0)

    # 创建小车主体和轮子的图形对象
    robot_body = Polygon(robot.get_outline(), fill=False, edgecolor='r', linewidth=2)
    left_wheel, right_wheel, wheel_width, wheel_radius = robot.get_wheels()
    left_wheel = Circle(left_wheel, wheel_radius, fill=True, color='black')
    right_wheel = Circle(right_wheel, wheel_radius, fill=True, color='black')

    ax.add_patch(robot_body)
    ax.add_patch(left_wheel)
    ax.add_patch(right_wheel)

    actual_path = [path[0]]

    # 创建用于显示实际路径的线条对象
    actual_line, = ax.plot([], [], 'r-', alpha=0.7, label=f'Actual Path ({selected_controller})')

    dt = 0.1  # 时间步长

    if selected_controller == 'PID':
        controller = PIDController(kp_linear=1.0, ki_linear=0.01, kd_linear=0.1,
                                   kp_angular=3.0, ki_angular=0.01, kd_angular=0.1,
                                   goal_tolerance=0.1)  # 无摩擦理论上ki项直接为0这里设置为一个小值
    elif selected_controller == 'Pure Pursuit':
        controller = PurePursuitController(lookahead_distance=1.0, linear_velocity=0.5, goal_tolerance=0.1)
    else:
        print("Invalid controller selected.")
        return

    goal_reached = False
    current_target_index = 1  # 从路径的第二个点开始
    cumulative_error = 0
    total_steps = 0

    while not goal_reached:
        current_target = path[current_target_index]

        if selected_controller == 'PID':
            v, omega = controller.control(np.array([robot.x, robot.y, robot.theta]), np.array(current_target))
            print(f"Current: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f}), Target: {current_target}, Control: v={v:.2f}, omega={omega:.2f}")
        elif selected_controller == 'Pure Pursuit':
            v, omega, current_target = controller.control(path, np.array([robot.x, robot.y, robot.theta]))
            print(f"Current: ({robot.x:.2f}, {robot.y:.2f}, {robot.theta:.2f}), Target: {current_target}, Control: v={v:.2f}, omega={omega:.2f}")

        # 根据控制器获得控制参数，作为前向运动输入
        robot.update_pose(v, omega, dt)

        # 追踪误差统计
        errors = [np.linalg.norm(np.array([robot.x, robot.y]) - np.array(p)) for p in path]
        min_error = min(errors)
        cumulative_error += min_error
        total_steps += 1

        # 更新小车图形
        robot_body.set_xy(robot.get_outline())
        left_wheel_pos, right_wheel_pos, _, _ = robot.get_wheels()
        left_wheel.center = left_wheel_pos
        right_wheel.center = right_wheel_pos

        actual_path.append((robot.x, robot.y))

        # 更新实际路径的线条
        actual_xs, actual_ys = zip(*actual_path)
        actual_line.set_data(actual_xs, actual_ys)

        fig.canvas.draw_idle()
        plt.pause(0.1)  # 调整暂停时间以控制模拟速度

        # 检查是否到达当前目标点
        if selected_controller == 'PID':
            if np.linalg.norm(np.array([robot.x, robot.y]) - np.array(current_target[:2])) < controller.goal_tolerance:
                current_target_index += 1
                if current_target_index >= len(path):
                    goal_reached = True
                    print("Goal reached!")

        if selected_controller == 'Pure Pursuit':
            if v <= 0.0001 and omega <= 0.0001:
                goal_reached = True
                print("Goal reached!")

    # 清除小车图形
    robot_body.remove()
    left_wheel.remove()
    right_wheel.remove()

    # 计算平均追踪误差
    average_tracking_error = cumulative_error / total_steps if total_steps > 0 else 0

    # 保存追踪结果
    save_tracking_results(map_id, algorithm, selected_controller, average_tracking_error, cumulative_error, total_steps)

    plt.legend()  # 更新图例
    fig.canvas.draw()


# 添加一个按钮来模拟路径
button_simulate = Button(plt.axes([0.75, 0.8, 0.15, 0.05]), 'Simulate')
button_simulate.on_clicked(simulate)

# 将confirm函数连接到确认按钮
button_confirm = Button(plt.axes([0.75, 0.6, 0.15, 0.05]), 'Confirm')
button_confirm.on_clicked(confirm)

# 将reset函数连接到重置按钮
button_reset = Button(plt.axes([0.75, 0.7, 0.15, 0.05]), 'Reset')
button_reset.on_clicked(reset)

# 创建用于平滑选项的CheckButtons
smooth_button = CheckButtons(plt.axes([0.75, 0.45, 0.15, 0.05]), ['smooth'])

fig.canvas.mpl_connect('button_press_event', onclick)

map_id = None
# 显示绘图
plt.show()
