#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_N 100  // 最大地图大小
#define INF 99999  // 无限大值

// 定义节点结构
typedef struct Node {
    int x, y;           // 坐标
    int f, g, h;        // f = g + h
    int parent_x;       // 父节点坐标
    int parent_y;
} Node;

// 定义优先队列(开放列表)
typedef struct {
    Node* nodes[MAX_N * MAX_N];
    int size;
} PriorityQueue;

// 初始化优先队列
void initQueue(PriorityQueue* queue) {
    queue->size = 0;
}

// 向优先队列中添加节点
void push(PriorityQueue* queue, Node* node) {
    int i = queue->size++;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (queue->nodes[parent]->f <= node->f)
            break;
        queue->nodes[i] = queue->nodes[parent];
        i = parent;
    }
    queue->nodes[i] = node;
}

// 从优先队列中取出f值最小的节点
Node* pop(PriorityQueue* queue) {
    Node* min = queue->nodes[0];
    Node* last = queue->nodes[--queue->size];
    int i = 0;
    while (i * 2 + 1 < queue->size) {
        int child = i * 2 + 1;
        if (child + 1 < queue->size &&
            queue->nodes[child + 1]->f < queue->nodes[child]->f)
            child++;
        if (last->f <= queue->nodes[child]->f)
            break;
        queue->nodes[i] = queue->nodes[child];
        i = child;
    }
    queue->nodes[i] = last;
    return min;
}

// 计算两点间的曼哈顿距离
int heuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

// 判断坐标是否有效
int isValid(int x, int y, int rows, int cols) {
    return (x >= 0 && x < rows && y >= 0 && y < cols);
}

// 判断是否为障碍物
int isUnBlocked(int grid[MAX_N][MAX_N], int x, int y) {
    return (grid[x][y] == 0);
}

// 打印地图（包含路径）
void printMap(int grid[MAX_N][MAX_N], int path[MAX_N][MAX_N], int rows, int cols,
             int start_x, int start_y, int end_x, int end_y) {
    printf("\n");
    // 打印列号
    printf("   ");
    for (int j = 0; j < cols; j++) {
        printf("%2d ", j);
    }
    printf("\n");

    for (int i = 0; i < rows; i++) {
        // 打印行号
        printf("%2d ", i);
        for (int j = 0; j < cols; j++) {
            if (i == start_x && j == start_y)
                printf("S  ");  // 起点
            else if (i == end_x && j == end_y)
                printf("E  ");  // 终点
            else if (grid[i][j] == 1)
                printf("██ ");  // 障碍物
            else if (path[i][j] == 1)
                printf("◆  ");  // 路径
            else
                printf("□  ");  // 可通行区域
        }
        printf("\n");
    }
    printf("\n图例: S=起点 E=终点 ██=障碍物 ◆=路径 □=可通行\n\n");
}

// A*算法主体
void astar(int grid[MAX_N][MAX_N], int rows, int cols,
          int start_x, int start_y, int end_x, int end_y) {
    // 判断起点和终点是否有效
    if (!isValid(start_x, start_y, rows, cols) ||
        !isValid(end_x, end_y, rows, cols)) {
        printf("起点或终点无效!\n");
        return;
    }

    // 判断起点和终点是否可达
    if (!isUnBlocked(grid, start_x, start_y) ||
        !isUnBlocked(grid, end_x, end_y)) {
        printf("起点或终点为障碍物!\n");
        return;
    }

    // 初始化访问数组
    int closed[MAX_N][MAX_N] = {0};

    // 初始化节点数组
    Node* nodes[MAX_N][MAX_N];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nodes[i][j] = (Node*)malloc(sizeof(Node));
            nodes[i][j]->x = i;
            nodes[i][j]->y = j;
            nodes[i][j]->f = INF;
            nodes[i][j]->g = INF;
            nodes[i][j]->h = INF;
            nodes[i][j]->parent_x = -1;
            nodes[i][j]->parent_y = -1;
        }
    }

    // 初始化起点
    int x = start_x, y = start_y;
    nodes[x][y]->f = 0;
    nodes[x][y]->g = 0;
    nodes[x][y]->h = 0;

    // 初始化开放列表
    PriorityQueue openList;
    initQueue(&openList);
    push(&openList, nodes[x][y]);

    // 定义8个方向的移动
    int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

    // 主循环
    int found_dest = 0;
    while (openList.size > 0) {
        Node* current = pop(&openList);
        x = current->x;
        y = current->y;
        closed[x][y] = 1;

        // 检查8个相邻节点
        for (int i = 0; i < 8; i++) {
            int new_x = x + dx[i];
            int new_y = y + dy[i];

            if (isValid(new_x, new_y, rows, cols)) {
                if (new_x == end_x && new_y == end_y) {
                    nodes[new_x][new_y]->parent_x = x;
                    nodes[new_x][new_y]->parent_y = y;
                    found_dest = 1;
                    printf("找到路径!\n");

                    // 创建路径数组并初始化
                    int path[MAX_N][MAX_N] = {0};

                    // 回溯并记录路径
                    int path_x = end_x, path_y = end_y;
                    int path_length = 0;
                    while (!(path_x == start_x && path_y == start_y)) {
                        path[path_x][path_y] = 1;
                        int temp_x = nodes[path_x][path_y]->parent_x;
                        int temp_y = nodes[path_x][path_y]->parent_y;
                        path_x = temp_x;
                        path_y = temp_y;
                        path_length++;
                    }

                    // 打印可视化地图
                    printf("\n最短路径长度: %d\n", path_length);
                    printMap(grid, path, rows, cols, start_x, start_y, end_x, end_y);
                    return;
                }
                else if (!closed[new_x][new_y] &&
                         isUnBlocked(grid, new_x, new_y)) {
                    int new_g = nodes[x][y]->g +
                               ((i < 4) ? 10 : 14); // 直线代价10，斜线代价14
                    int new_h = heuristic(new_x, new_y, end_x, end_y) * 10;
                    int new_f = new_g + new_h;

                    if (nodes[new_x][new_y]->f == INF ||
                        nodes[new_x][new_y]->f > new_f) {
                        nodes[new_x][new_y]->f = new_f;
                        nodes[new_x][new_y]->g = new_g;
                        nodes[new_x][new_y]->h = new_h;
                        nodes[new_x][new_y]->parent_x = x;
                        nodes[new_x][new_y]->parent_y = y;
                        push(&openList, nodes[new_x][new_y]);
                    }
                }
            }
        }
    }

    if (!found_dest) {
        printf("无法找到路径!\n");
    }

    // 释放内存
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            free(nodes[i][j]);
        }
    }
}

// 随机生成地图
void generateMap(int grid[MAX_N][MAX_N], int rows, int cols, float obstacle_density) {
    // 初始化随机数生成器
    srand(time(NULL));

    // 先将地图全部置为0
    memset(grid, 0, sizeof(int) * MAX_N * MAX_N);

    // 根据密度随机放置障碍物
    int total_obstacles = (int)(rows * cols * obstacle_density);
    while (total_obstacles > 0) {
        int x = rand() % rows;
        int y = rand() % cols;
        // 确保起点和终点不会被放置障碍物
        if (grid[x][y] == 0 && !(x == 0 && y == 0) && !(x == rows-1 && y == cols-1)) {
            grid[x][y] = 1;
            total_obstacles--;
        }
    }
}

// 主函数示例
int main() {
    int grid[MAX_N][MAX_N];

    int rows = 15, cols = 15;  // 更大的地图尺寸
    int start_x = 0, start_y = 0;
    int end_x = rows-1, end_y = cols-1;
    float obstacle_density = 0.5;  // 50%的障碍物密度

    // 生成随机地图
    generateMap(grid, rows, cols, obstacle_density);

    printf("随机生成的地图:\n");
    int empty_path[MAX_N][MAX_N] = {0};  // 用于初始显示
    printMap(grid, empty_path, rows, cols, start_x, start_y, end_x, end_y);

    printf("\n寻找从(%d,%d)到(%d,%d)的路径:\n",
           start_x, start_y, end_x, end_y);
    astar(grid, rows, cols, start_x, start_y, end_x, end_y);

    return 0;
}
