import numpy as np
import pandas as pd
import os
import graphviz


# 定义节点类
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


# 计算基尼系数
def gini(y):
    unique_classes = np.unique(y)
    n_instances = len(y)
    gini_index = 0.0
    for cls in unique_classes:
        p_cls = np.sum(y == cls) / n_instances
        gini_index += p_cls * (1.0 - p_cls)
    return gini_index


# 拆分数据集
def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] < threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


# 选择最佳拆分点
def find_best_split(X, y):
    n_features = X.shape[1]
    best_gini = float('inf')
    best_feature_index = -1
    best_threshold = 0
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            gini_index = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
            if gini_index < best_gini:
                best_gini = gini_index
                best_feature_index = feature_index
                best_threshold = threshold
    return best_feature_index, best_threshold


# 构建决策树（预剪枝）
def build_tree_pruned(X, y, depth=0, max_depth=5, min_samples_split=2):
    # 若样本数小于等于阈值或者达到最大深度，则停止分裂，返回叶节点
    if len(y) <= min_samples_split or depth >= max_depth or len(np.unique(y)) == 1:
        return Node(predicted_class=np.bincount(y).argmax())
    
    feature_index, threshold = find_best_split(X, y)
    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
    
    # 如果分裂后的子集样本数小于等于阈值，则停止分裂，返回叶节点
    if len(y_left) <= min_samples_split or len(y_right) <= min_samples_split:
        return Node(predicted_class=np.bincount(y).argmax())
    
    node = Node(predicted_class=None)
    node.feature_index = feature_index
    node.threshold = threshold
    node.left = build_tree_pruned(X_left, y_left, depth + 1, max_depth, min_samples_split)
    node.right = build_tree_pruned(X_right, y_right, depth + 1, max_depth, min_samples_split)
    return node


# 预测单个样本
def predict_sample(tree, x):
    while tree.left:
        if x[tree.feature_index] < tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.predicted_class


# 预测多个样本
def predict_tree(tree, X):
    return [predict_sample(tree, x) for x in X]


# 加载数据集
column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

train_data = pd.read_csv('D:/大三课程/机器学习/实验/实验七/adult_train.txt',
                         names=column_names, na_values=' ?', delimiter=',')
test_data = pd.read_csv('D:/大三课程/机器学习/实验/实验七/adult_test.txt',
                        names=column_names, na_values=' ?', delimiter=',')

# 删除缺失值和'native-country'属性
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
train_data.drop(columns=['native-country'], inplace=True)
train_data.drop(columns=['fnlwgt'], inplace=True)
test_data.drop(columns=['fnlwgt'], inplace=True)
test_data.drop(columns=['native-country'], inplace=True)

# 将标签编码为0和1
train_data['income'] = train_data['income'].map({' <=50K': 0, ' >50K': 1})
test_data['income'] = test_data['income'].map({' <=50K.': 0, ' >50K.': 1})

# 合并训练集和测试集
combined_data = pd.concat([train_data, test_data], ignore_index=True)
# 独热编码
combined_data = pd.get_dummies(combined_data, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'])

# 提取特征和标签
y_combined = combined_data['income']
X_combined = combined_data.drop(columns=['income'])

# 将分类变量映射为整数编码
for column in X_combined.select_dtypes(include=['object']).columns:
    X_combined[column] = X_combined[column].astype('category').cat.codes

# 划分回训练集和测试集
X_train = X_combined.iloc[:len(train_data), :]
X_test = X_combined.iloc[len(train_data):, :]

y_train = train_data['income']
y_test = test_data['income']

feature_names = X_combined.columns

# 设置Graphviz的路径
os.environ["PATH"] += os.pathsep + r'E:/下载中专/windows_10_cmake_Release_Graphviz-10.0.1-win64/Graphviz-10.0.1-win64/bin'

# # 设置不同的参数组合
# max_depths = [5, 10, 15, 20]  # 不同的最大深度
# min_samples_splits = [10, 50, 100, 200]  # 不同的最小分裂样本数

# # 用于记录不同参数组合的准确率
# results = defaultdict(dict)

# for max_depth in max_depths:
#     for min_samples_split in min_samples_splits:
#         # 构建决策树
#         tree_pruned = build_tree_pruned(X_train.values, y_train.values, max_depth=max_depth, min_samples_split=min_samples_split)
        
#         y_pred = predict_tree(tree_pruned, X_test.values)
#         accuracy = np.mean(y_pred == y_test)
        
#         # 记录结果
#         results[max_depth][min_samples_split] = accuracy

# # 打印结果
# print("Results:")
# for max_depth, splits in results.items():
#     for min_samples_split, accuracy in splits.items():
#         print(f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Accuracy: {accuracy}")

# 确定最优组合后以最优组合建立决策树
tree_pruned = build_tree_pruned(X_train.values, y_train.values, max_depth=10, min_samples_split=10)
y_pred = predict_tree(tree_pruned, X_test.values)
accuracy = np.mean(y_pred == y_test)
print("预剪枝后准确率:", accuracy)


# 将决策树保存为.dot文件
def export_tree_as_dot(tree, filename):
    with open(filename, 'w') as f:
        f.write('digraph DecisionTree {\n')
        export_tree_as_dot_rec(tree, f)
        f.write('}')


def export_tree_as_dot_rec(tree, f, node_id='root'):
    if tree.left is None and tree.right is None:
        f.write(f'{node_id} [label="{tree.predicted_class}"];\n')
    else:
        f.write(f'{node_id} [label="{feature_names[tree.feature_index]} <= {tree.threshold}"];\n')
        left_id = f'{node_id}_left'
        right_id = f'{node_id}_right'
        export_tree_as_dot_rec(tree.left, f, left_id)
        export_tree_as_dot_rec(tree.right, f, right_id)
        f.write(f'{node_id} -> {left_id} [label="True"];\n')
        f.write(f'{node_id} -> {right_id} [label="False"];\n')


export_tree_as_dot(tree_pruned, 'decision_tree_pruned.dot')

# 使用Graphviz库生成图像并显示
with open('decision_tree_pruned.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).view()
