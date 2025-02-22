#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

// 交换两个元素
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// 1. 冒泡排序
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
            }
        }
    }
}

// 2. 优化的冒泡排序
void improvedBubbleSort(int arr[], int n) {
    int swapped;
    for (int i = 0; i < n-1; i++) {
        swapped = 0;
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

// 3. 选择排序
void selectionSort(int arr[], int n) {
    int min_idx;
    for (int i = 0; i < n-1; i++) {
        min_idx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            swap(&arr[i], &arr[min_idx]);
        }
    }
}

// 4. 插入排序
void insertionSort(int arr[], int n) {
    int key, j;
    for (int i = 1; i < n; i++) {
        key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

// 5. 希尔排序
void shellSort(int arr[], int n) {
    for (int gap = n/2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j-gap] > temp; j -= gap) {
                arr[j] = arr[j-gap];
            }
            arr[j] = temp;
        }
    }
}

// 6. 归并排序的合并函数
void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++) L[i] = arr[left + i];
    for (j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// 6. 归并排序
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// 7. 快速排序的分区函数
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

// 7. 快速排序
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// 8. 堆排序的堆化函数
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

// 8. 堆排序
void heapSort(int arr[], int n) {
    // 构建最大堆
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // 一个个从堆中取出元素
    for (int i = n - 1; i > 0; i--) {
        swap(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

// 9. 计数排序
void countingSort(int arr[], int n) {
    // 找到最大和最小值
    int max = arr[0], min = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
        if (arr[i] < min) min = arr[i];
    }

    // 计算范围
    int range = max - min + 1;

    // 分配计数数组和输出数组
    int* count = (int*)calloc(range, sizeof(int));
    int* output = (int*)malloc(n * sizeof(int));

    if (count == NULL || output == NULL) {
        printf("内存分配失败！\n");
        return;
    }

    // 统计每个元素出现的次数
    for (int i = 0; i < n; i++) {
        count[arr[i] - min]++;
    }

    // 累加统计数组
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }

    // 构建输出数组
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i] - min] - 1] = arr[i];
        count[arr[i] - min]--;
    }

    // 将排序结果复制回原数组
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(count);
    free(output);
}

// 生成随机数组
void generateRandomArray(int arr[], int n, int max_value) {
    for(int i = 0; i < n; i++) {
        arr[i] = rand() % max_value;
    }
}

// 复制数组
void copyArray(int src[], int dest[], int n) {
    memcpy(dest, src, n * sizeof(int));
}

// 验证排序是否正确
int isSorted(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i-1]) return 0;
    }
    return 1;
}

// 打印数组
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}

// 定义排序函数指针类型
typedef void (*SortFunction)(int[], int);

// 定义排序算法结构体
typedef struct {
    char* name;
    SortFunction function;
    int needs_size_only;  // 1表示只需要size参数，0表示需要其他参数
} SortAlgorithm;

// 包装函数，使归并排序和快速排序接口统一
void mergeSort_wrapper(int arr[], int n) {
    mergeSort(arr, 0, n-1);
}

void quickSort_wrapper(int arr[], int n) {
    quickSort(arr, 0, n-1);
}

// 测试排序算法性能
void testSortingAlgorithm(int original[], int n, SortAlgorithm algo) {
    int* arr = (int*)malloc(n * sizeof(int));
    if (arr == NULL) {
        printf("内存分配失败！\n");
        return;
    }
    copyArray(original, arr, n);

    // 执行排序并计时
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    algo.function(arr, n);

    clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算时间（纳秒转换为毫秒）
    double time_taken = (end.tv_sec - start.tv_sec) * 1000.0 +
                       (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("%-20s | 时间: %8.2f ms | 正确性: %s\n",
           algo.name,
           time_taken,
           isSorted(arr, n) ? "通过" : "失败");

    free(arr);
}

int main() {
    // 设置随机数种子
    srand(time(NULL));

    // 定义所有排序算法
    SortAlgorithm algorithms[] = {
        {"冒泡排序", bubbleSort, 1},
        {"优化冒泡排序", improvedBubbleSort, 1},
        {"选择排序", selectionSort, 1},
        {"插入排序", insertionSort, 1},
        {"希尔排序", shellSort, 1},
        {"归并排序", mergeSort_wrapper, 1},
        {"快速排序", quickSort_wrapper, 1},
        {"堆排序", heapSort, 1},
        {"计数排序", countingSort, 1}
    };
    int num_algorithms = sizeof(algorithms) / sizeof(algorithms[0]);

    // 测试不同数据规模
    int sizes[] = {10000, 50000, 200000, 1000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        printf("\n=== 测试数据规模: %d ===\n", n);

        // 生成随机数组
        int* original = (int*)malloc(n * sizeof(int));
        generateRandomArray(original, n, 10000);  // 生成0-9999的随机数

        // 测试所有排序算法
        for (int i = 0; i < num_algorithms; i++) {
            testSortingAlgorithm(original, n, algorithms[i]);
        }

        free(original);
    }

    return 0;
}
