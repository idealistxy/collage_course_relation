import pandas as pd
import numpy as np
import graphviz
import os


class TreeNode:
    def __init__(self, feature_index=None, value=None, value_name=None):
        self.feature_index = feature_index  # 子节点划分属性
        self.value_name = value_name        # 节点属性名
        self.value = value                  # 叶子节点的预测值
        self.children = {}                  # 子节点字典，key为特征取值，value为子节点

    def export_tree_as_dot(self, filename):
        dot_string = self._generate_dot_string()
        with open(filename, 'w') as f:
            f.write(dot_string)  

    def _generate_dot_string(self, parent_id=None, node_id=0):
        if parent_id is None:
            dot_string = "digraph Tree {\n"
        else:
            dot_string = "" 

        if self.feature_index is not None:  
            dot_string += f"{parent_id} -> {node_id} ;\n"  
            if self.value_name:
                dot_string += f"{node_id} [label=\"{feature_names[self.value_name]} = {self.value}\"];\n"  
        else:
            dot_string += f"{parent_id} -> {node_id} ;\n" 
            dot_string += f"{node_id} [label=\"Class: {self.value}\"];\n"  

        for value, child_node in self.children.items():  
            dot_string += child_node._generate_dot_string(parent_id=node_id, node_id=node_id * 2 + value)  

        if parent_id is None:  
            dot_string += "}" 

        return dot_string


def calculate_gini(y):
    classes = np.unique(y)
    gini = 1
    total_samples = len(y)
    for c in classes:
        p = np.sum(y == c) / total_samples
        gini -= p ** 2
    return gini


def find_best_split(X, y):
    best_gini = float('inf')
    best_feature_index = None
    flag = False
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = np.where(X[:, feature_index] <= threshold)[0]
            right_indices = np.where(X[:, feature_index] > threshold)[0]
            gini = (len(left_indices) / len(y)) * calculate_gini(y[left_indices]) + \
                   (len(right_indices) / len(y)) * calculate_gini(y[right_indices])

            if gini < best_gini:
                best_gini = gini
                best_feature_index = feature_index
                flag = True
    if flag:
        print(f'划分前数据集行数{len(y)}, 选中划分属性：{feature_names[best_feature_index]}, 划分后节点数：{len(thresholds)}')
    return best_feature_index


def build_tree(X, y, max_depth=None, va=None, before_feature_indices=None):
    if max_depth == 0 or len(np.unique(y)) == 1:
        return TreeNode(value=np.bincount(y).argmax())

    feature_index = find_best_split(X, y)

    if feature_index is None:
        return TreeNode(value=np.bincount(y).argmax())

    node = TreeNode(feature_index=feature_index, value=va, value_name=before_feature_indices)
    for value in np.unique(X[:, feature_index]):
        indices = np.where(X[:, feature_index] == value)[0]
        node.children[value] = build_tree(X[indices], y[indices], max_depth=max_depth-1, va=value, before_feature_indices=feature_index)

    return node


def predict_tree(node, x):
    if node is None:
        return 0
    if node.children:
        value = x[node.feature_index]
        if value in node.children:
            return predict_tree(node.children[value], x)
    return node.value


def prune_tree(node, X_val, y_val):
    if not node.children:
        return node.value

    for value, child_node in node.children.items():
        node.children[value] = prune_tree(child_node, X_val, y_val)

    y_pred = [predict_tree(node, x) for x in X_val]
    accuracy_before_prune = np.mean(y_pred == y_val)

    node_is_leaf = True
    for child_node in node.children.values():
        if isinstance(child_node, TreeNode):
            node_is_leaf = False
            break

    if node_is_leaf:
        return node.value  # Return the leaf node prediction directly

    node_value = np.argmax(np.bincount(y_val))
    node_accuracy_after_prune = np.mean([node_value] * len(X_val) == y_val)

    if node_accuracy_after_prune >= accuracy_before_prune:
        print(f'Pruning node with feature index {node.feature_index}')
        return node_value
    else:
        return node  # Return the original node if pruning is not beneficial


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
X_train = X_combined.iloc[:len(train_data)]
X_test = X_combined.iloc[len(train_data):]

y_train = train_data['income']
y_test = test_data['income']

feature_names = X_combined.columns


# # 设置Graphviz的路径
os.environ["PATH"] += os.pathsep + r'E:/下载中专/windows_10_cmake_Release_Graphviz-10.0.1-win64/Graphviz-10.0.1-win64/bin'

# 使用示例
tree = build_tree(X_train.values, y_train.values, max_depth=3)
y_pred = [predict_tree(tree, x) for x in X_test.values]
accuracy = np.mean(y_pred == y_test)
print("剪枝前准确率:", accuracy)

# tree.export_tree_as_dot('decision_tree.dot')

# # 使用Graphviz库生成图像并显示
# with open('decision_tree.dot') as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph).view()


# Example usage of pruning
pruned_tree = prune_tree(tree, X_test.values, y_test.values)
y_pruned_pred = [predict_tree(pruned_tree, x) for x in X_test.values]
accuracy_after_prune = np.mean(y_pruned_pred == y_test)
print("Accuracy after pruning:", accuracy_after_prune)
pruned_tree.export_tree_as_dot('pruned_decision_tree.dot')

# 使用Graphviz库生成图像并显示
with open('pruned_decision_tree.dot') as f:
    pruned_dot_graph = f.read()
graphviz.Source(pruned_dot_graph).view()
