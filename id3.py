"""
A Python implementation of the ID3 decision tree algorithm including pruning strategy.
Author: Yanming Shao, Baitong Lu
Environment: Python 3.11
"""


import pandas as pd
import math


# Define the class of decision tree nodes
class Node:
    def __init__(self):
        self.attribute = None
        self.children = {}
        self.value = None

# Compute entropy
def entropy(data: pd.DataFrame, target_attr: str):
    # Count the frequency of target attributes
    value_counts = data[target_attr].value_counts()
    entropy = 0
    total_records = len(data)
    # Calculate the probability of each category and calculate the entropy
    for value in value_counts:
        probability = value / total_records
        entropy += -probability * math.log2(probability)
    return entropy

# Calculate information gain
def information_gain(data: pd.DataFrame, attribute: str, target_attr: str):
    # Calculate overall entropy
    total_entropy = entropy(data, target_attr)
    # Split the dataset
    attribute_values = data[attribute].unique()
    weighted_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset, target_attr)
    # Calculate information gain
    info_gain = total_entropy - weighted_entropy
    return info_gain

# Define a function to select the best attributes for splitting
def select_best_attribute(data: pd.DataFrame, attributes: list[str], target_attr: str):
    best_attribute = None
    max_info_gain = -1
    for attribute in attributes:
        gain = information_gain(data, attribute, target_attr)
        if gain > max_info_gain:
            max_info_gain = gain
            best_attribute = attribute
    return best_attribute

# Build a decision tree and add maximum depth pruning
def build_decision_tree(data: pd.DataFrame, attributes: list[str], target_attr: str, max_depth: int):
    # If the maximum depth is reached or the dataset contains only one category, return leaf nodes
    if max_depth == 0 or len(data[target_attr].unique()) == 1:
        leaf = Node()
        leaf.value = data[target_attr].iloc[0]
        return leaf
    # If there are no available attributes, return the leaf node, and the category is the category that appears most in the data set.
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

# When processing unlearned queries, return the category or most categories of a decision tree node
def most_attribute(tree_node: Node):
    # If the node is a leaf node, return the category of the leaf node
    if not tree_node.children:
        return tree_node.value
    # If the node is not a leaf node, find the category that appears most in the child node
    child_values = [most_attribute(child) for child in tree_node.children.values()]
    return max(child_values, key=child_values.count)

# Prediction function
def predict(tree_node: Node, data: pd.Series):
    if tree_node.value is not None:
        return tree_node.value
    attribute_value = data[tree_node.attribute]
    if attribute_value in tree_node.children:
        return predict(tree_node.children[attribute_value], data)
    else:
        return most_attribute(tree_node)

# Main function
def main():
    #Dataset reading
    train_data = pd.read_csv('./data/train.csv')
    predict_data = pd.read_csv('./data/predict.csv')

    # Build decision tree
    attributes = train_data.columns[:-1].tolist()
    target_attribute = 'weather'

    # Build a decision tree and apply maximum depth pruning
    max_depth = 5
    decision_tree = build_decision_tree(train_data, attributes, target_attribute, max_depth)

    # Predict and output the results
    predicted_results = []
    for _, row in predict_data.iterrows():
        prediction = predict(decision_tree, row)
        predicted_results.append(prediction)

    # Store prediction results
    predict_data['weather'] = predicted_results
    predict_data.to_csv('./data/result.csv', index=False)


if __name__ == '__main__':
    main()
