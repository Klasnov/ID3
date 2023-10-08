import pandas as pd
import math

# 数据集读取
train_data = pd.read_csv('./data/train.csv')
predict_data = pd.read_csv('./data/predict.csv')

# 定义一个函数计算熵
def entropy(data, target_attr):
    # 统计目标属性的频数
    value_counts = data[target_attr].value_counts()
    entropy = 0
    total_records = len(data)
    # 计算每个类别的概率并计算熵
    for value in value_counts:
        probability = value / total_records
        entropy += -probability * math.log2(probability)
    return entropy

# 定义一个函数计算信息增益
def information_gain(data, attribute, target_attr):
    # 计算总体熵
    total_entropy = entropy(data, target_attr)
    # 分割数据集
    attribute_values = data[attribute].unique()
    weighted_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset, target_attr)
    # 计算信息增益
    info_gain = total_entropy - weighted_entropy
    return info_gain

# 定义一个函数用于选择最佳属性进行分裂
def select_best_attribute(data, attributes, target_attr):
    best_attribute = None
    max_info_gain = -1
    for attribute in attributes:
        gain = information_gain(data, attribute, target_attr)
        if gain > max_info_gain:
            max_info_gain = gain
            best_attribute = attribute
    return best_attribute

# 定义决策树节点的类
class Node:
    def __init__(self):
        self.attribute = None
        self.children = {}
        self.value = None

# 构建决策树，添加最大深度剪枝
def build_decision_tree(data, attributes, target_attr, max_depth):
    # 如果达到最大深度或数据集只包含一个类别，则返回叶节点
    if max_depth == 0 or len(data[target_attr].unique()) == 1:
        leaf = Node()
        leaf.value = data[target_attr].iloc[0]
        return leaf
    # 如果没有可用属性，返回叶节点，类别为数据集中出现最多的类别
    if len(attributes) == 0:
        leaf = Node()
        leaf.value = data[target_attr].value_counts().idxmax()
        return leaf
    best_attribute = select_best_attribute(data, attributes, target_attr)
    tree = Node()
    tree.attribute = best_attribute
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if len(subset) == 0:
            leaf = Node()
            leaf.value = data[target_attr].value_counts().idxmax()
            tree.children[value] = leaf
        else:
            new_attributes = attributes.copy()
            new_attributes.remove(best_attribute)
            tree.children[value] = build_decision_tree(subset, new_attributes, target_attr, max_depth - 1)
    return tree

# 预测函数
def predict(tree, data):
    if tree.value is not None:
        return tree.value
    attribute_value = data[tree.attribute]
    if attribute_value in tree.children:
        return predict(tree.children[attribute_value], data)
    else:
        return data['weather'].value_counts().idxmax()

# 构建决策树
attributes = train_data.columns[:-1].tolist()
target_attribute = 'weather'

# 构建决策树并应用最大深度剪枝
max_depth = 5  # 你可以根据需要调整最大深度
decision_tree = build_decision_tree(train_data, attributes, target_attribute, max_depth)

# 预测并输出结果
predicted_results = []
for index, row in predict_data.iterrows():
    prediction = predict(decision_tree, row)
    predicted_results.append(prediction)

# 存储预测结果
predict_data['weather'] = predicted_results
predict_data.to_csv('./data/result.csv', index=False)
