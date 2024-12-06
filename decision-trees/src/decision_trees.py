# Intelligent Systems - Decision Trees
# Designed and developed by Kobi Chambers - Griffith University

# Import necessary relevent packages and modules
try:
    import os
    import sys
    import random
    import pandas as pd
    import numpy as np
    # import matplotlib.pyplot as plt
    import tkinter as tk
    from tkinter import filedialog
    from collections import Counter

except ImportError as e:
    print(f"Error importing module: {e}")
    print(f"Please ensure that module is installed...")
    sys.exit(1)


class DecisionTree:
    """
    Class to represent a decision tree
    """

    def __init__(self):
        self.root = None

    class Node:
        """
        Class to represent the nodes in a decision tree
        """

        def __init__(self, feature=None, value=None, leaf=None, true_branch=None, false_branch=None):
            self.feature = feature  # The feature to split on
            self.value = value  # The value of the feature to make the split
            self.leaf = leaf  # The predicted label if the node is a leaf
            self.true_branch = true_branch  # The next node if the condition is true
            self.false_branch = false_branch  # The next node if the condition is false

    def calculate_entropy(self, target):
        """
        Implementation of calculate_entropy function
        """
        _, counts = np.unique(target, return_counts=True)
        entropy = 0.0

        for count in counts:
            proportion = count / len(target)
            entropy -= proportion * np.log2(proportion)

        return entropy

    def calculate_gain(self, parent_impurity, true_impurity, false_impurity, true_weight, false_weight):
        """
        Calculate the information gain based on the impurity measure
        """
        # Subtract the weighted impurities from the parent impurity
        parent_weight = true_weight + false_weight
        gain = parent_impurity - ((true_weight / parent_weight) * true_impurity) - (
            (false_weight / parent_weight) * false_impurity)

        return gain

    def get_majority_label(self, labels):
        """
        Find which label is most common
        """
        label_counts = Counter(labels)
        majority_label = label_counts.most_common(1)[0][0]

        return majority_label

    def find_best_split(self, data, target, features):
        """
        Implementation of find_best_split function
        """
        best_gain = -float('inf')
        best_feature = None
        best_value = None
        data.reset_index(drop=True, inplace=True)
        target = data[target]
        print(data)

        for feature in features:
            # Iterate over all possible values for the current feature
            values = np.unique(data[feature])

            for value in values:
                # Split the data into two groups based on the current feature and value
                true_data = np.where(data[feature] == value)[0]
                false_data = np.where(data[feature] != value)[0]

                # Get target labels for true and false data
                true_target = target[true_data]
                false_target = target[false_data]

                # Calculate the impurity of the split
                parent_impurity = self.calculate_entropy(target)
                true_impurity = self.calculate_entropy(true_target)
                false_impurity = self.calculate_entropy(false_target)

                # Calculate the information gain
                gain = self.calculate_gain(parent_impurity, true_impurity, false_impurity, len(
                    true_target), len(false_target))

                # Update the best split if the current gain is higher
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def build_tree(self, data, target, features, min_sample_split):
        """
        Function to build the decision tree
        """
        labels = data[target]

        # If all labels are the same, create a leaf node
        if len(np.unique(labels)) == 1:
            return self.Node(leaf=np.unique(labels)[0])

        # If current sample size is equal to or smaller than our decided min, create a leaf node
        if len(data) <= min_sample_split:
            return self.Node(leaf=self.get_majority_label(labels))

        # Find the best split
        best_feature, best_value = self.find_best_split(data, target, features)

        # Check if gain is 0
        if best_feature is None or best_value is None:
            return self.Node(leaf=self.get_majority_label(labels))

        # Split the data into two groups based on the best split
        true_data = data[data[best_feature] == best_value]
        false_data = data[data[best_feature] != best_value]

        # Recursive calls to build the true and false branches
        true_branch = self.build_tree(
            true_data, target, features, min_sample_split)
        false_branch = self.build_tree(
            false_data, target, features, min_sample_split)

        # Create and return the current node
        return self.Node(best_feature, best_value, true_branch, false_branch)

    def fit(self, data, target, features, min_sample_split):
        """
        Function to fit the decision tree with training data
        """
        self.root = self.build_tree(data, target, features, min_sample_split)

    def predict(self, node, sample):
        """
        Function to make predictions using the decision tree
        """
        if node.leaf is not None:
            return node.leaf

        if sample[node.feature] == node.value:
            return self.predict(node.true_branch, sample)

        else:
            return self.predict(node.false_branch, sample)


def process_input_file(input_file_path):
    """
    Function to process the input csv file
    Returns loaded dataframe, target column and feature columns
    """
    try:
        # Read csv into pandas dataframe
        df = pd.read_csv(input_file_path)

        # Drop rows with missing values if needed
        df = df.dropna()

        # Set target and features as the column names
        column_names = df.columns.tolist()
        target = column_names[0]
        features = column_names[1:]

    # Handle file error
    except:
        sys.exit("Error occured while opening the file. Closing program...")

    return df, target, features


def get_input_file(program_folder):
    """
    Function to get the input file path either from the program folder or from user
    Utilises process_input_file function, then returns df, target and features to driver code
    """
    # Create input filepath
    input_file_path = os.path.join(program_folder, 'votes.csv')

    # Check that filepath exists within program folder
    if os.path.exists(input_file_path):
        return process_input_file(input_file_path)

    # If votes.csv is not within program folder, open filedialog to allow the user to find the input file
    else:
        # Create a tk window and hide it
        root = tk.Tk()
        root.withdraw()

        # Get filepath from user, initial directory is within the same folder as program
        input_file_path = filedialog.askopenfilename(
            initialdir=program_folder,
            title="Please select the votes csv file",
            filetypes=(("CSV files", "*.csv"),)
        )

        # Check if user selected a file
        if input_file_path:
            return process_input_file(input_file_path)

        else:
            sys.exit("User closed file dialogue... exiting program.\n")


def split_data_randomly(data, target):
    """
    Function to randomly split the data into training and test data
    Returns the two separated data sets
    """
    # Set a seed for reproducibility while building and debugging
    # Comment out if we want random splits for each program exection
    random.seed(42)

    # Shuffle indices
    indices = data.index.tolist()
    random.shuffle(indices)

    # Define split proportions
    train_proportion = 0.8

    # Find the index of the split
    split_index = int(len(indices) * train_proportion)

    # Split by index
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Create the two datasets
    train_data = data.loc[train_indices].reset_index(drop=True)
    test_data = data.loc[test_indices].reset_index(drop=True)

    # Create labels
    train_labels = train_data[target]
    test_labels = test_data[target]

    return train_data, test_data, train_labels, test_labels


def get_accuracy(test_labels, predictions):
    """
    Function to find the accuracy of the dt model
    """
    ...


def get_confusion_matrix(test_labels, predictions):
    """
    Function to produce the confusion matrix for the dt model
    """
    ...


def get_classification_report(test_labels, predictions):
    """
    Function to produce a classification report on the dt model
    """
    ...


def main():
    """
    Function to execute driver code
    """
    # Get program folder name
    program_folder = os.path.dirname(os.path.abspath(__file__))

    # Get the dataframe, target and features from input csv file
    data, target, features = get_input_file(program_folder)

    if data is None:
        sys.exit("Data frame loaded is empty... exiting program.\n")

    # Create random split
    train_data, test_data, train_labels, test_labels = split_data_randomly(
        data, target)

    # Check if train_data and train_labels have the same indices
    if not train_data.index.equals(train_labels.index):
        sys.exit("train_data and train_labels are not aligned.")

    # Set the minimum samples to create a leaf node
    percentage_split = 0.05
    min_sample_split = percentage_split * len(train_data)

    # Build our tree, fitting with the training data
    tree = DecisionTree()
    tree.fit(train_data, target, features, min_sample_split)

    print(f'tree: {tree}')
    print(f'tree root: {tree.root}')

    # Make predictions using the trained decision tree model
    predictions = [tree.predict(tree.root, sample)
                   for _, sample in test_data.iterrows()]
    print(f'prediction[0]: {predictions[0]}')

    # Calculate accuracy and other metrics
    # accuracy = get_accuracy(test_labels, predictions)
    # confusion_matrix = get_confusion_matrix(test_labels, predictions)
    # classification_report = get_classification_report(test_labels, predictions)

    ...


if __name__ == '__main__':
    main()
