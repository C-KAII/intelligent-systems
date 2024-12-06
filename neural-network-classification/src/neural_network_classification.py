# Intelligent Systems - Neural Network Classification
# Designed and developed by Kobi Chambers - Griffith University

# Import necessary relevent packages and modules
try:
    import sys
    import pickle
    import time
    import numpy as np
    # import tkinter as tk
    # from tkinter import filedialog
    import matplotlib.pyplot as plt

except ImportError as e:
    print(f"Error importing module: {e}")
    print(f"Please ensure that module is installed...")
    sys.exit(1)

# Constants defined here can be manipulated for experimentation

SIZES = [
    3072, 30, 10
]

LEARNING_RATES = [
    0.001, 0.01, 0.1, 1.0, 10.0, 100.0
]

MINI_BATCH_SIZES = [
    1, 5, 20, 100, 300
]

EPOCHS = 20

DFT_LEARNING_RATE = 0.1

DFT_BATCH_SIZE = 100

##################################
### INPUT PROCESSING FUNCTIONS ###
##################################


def unpickle(file):
    """
    Function to unpickle the dataset batches.

    Args:
        file (str): The path to the pickle file.

    Returns:
        dict: The unpickled data dictionary.
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    return data_dict


def preprocess_data(data_dict):
    """
    Function to normalize pixel values to the range between 0 and 1.

    Args:
        data_dict (dict): The dictionary containing data and labels.

    Returns:
        tuple: A tuple containing the normalized data and labels.
    """
    data = data_dict[b'data'] / 255.0
    labels = data_dict[b'labels']

    return data, labels


def encode_labels(digit):
    """
    Function to perform one-hot encoding of labels.

    Args:
        digit (int): The label to be encoded.

    Returns:
        ndarray: The encoded label array.
    """
    label = np.zeros(10, dtype=int)
    label[digit] = 1

    return label

############################
### ACTIVATION FUNCTIONS ###
############################


def sig(x):
    """
    Helper function to calculate the sigmoid activation.

    Args:
        x (ndarray): The input array.

    Returns:
        ndarray: The output array after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def sig_deriv(x):
    """
    Helper function to calculate the derivative of the sigmoid activation.

    Args:
        x (ndarray): The input array.

    Returns:
        ndarray: The output array after applying the derivative of the sigmoid function.
    """
    s = sig(x)
    return s * (1 - s)

###############################
### NN CLASS IMPLEMENTATION ###
###############################


class FFNeuralNet():
    """
    Class implementation of a Feed-Forward Neural Network.
    This class builds our neural network,
    passes information in one direction through input neurons,
    and utilizes input, hidden, and output network layers.
    """

    def __init__(self, sizes, learning_rate=0.1, batch_size=100, epochs=20):
        """
        Initializes a new instance of the FFNeuralNet class.

        Args:
            sizes (list): The number of neurons in each layer of the network, stored in a list.
            learning_rate (float): The learning rate for gradient descent. Default is 0.1.
            batch_size (int): The size of mini-batches for training. Default is 100.
            epochs (int): The number of training epochs. Default is 20.
        """
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights, self.biases = self.initialize_parameters()
        self.activations = {}

    def initialize_parameters(self):
        """
        Initializes the weights and biases of the neural network.

        Returns:
            tuple: A tuple containing the initialized weights and biases.
        """
        # Get layer sizes
        sizes = self.sizes
        input_size = sizes[0]
        hidden_size = sizes[1]
        output_size = sizes[2]

        # Initialise weights dict with random values
        weights = {
            'W1': np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / hidden_size),
            'W2': np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / output_size)
        }

        # Initialise biases dict with random values
        biases = {
            'b1': np.random.randn(hidden_size) * np.sqrt(1.0 / hidden_size),
            'b2': np.random.randn(output_size) * np.sqrt(1.0 / output_size)
        }

        return weights, biases

    def feed_forward(self, x):
        """
        Performs the feed-forward process to calculate the output layer activations.

        Args:
            x (ndarray): The input array.

        Returns:
            ndarray: The activation value after applying the feed-forward process.
        """
        weights = self.weights
        biases = self.biases

        # Input layer activations
        self.activations['A0'] = x

        # Hidden layer activations
        self.activations['Z1'] = np.dot(
            weights['W1'], self.activations['A0']) + biases['b1']
        self.activations['A1'] = sig(self.activations['Z1'])

        # Output layer activations
        self.activations['Z2'] = np.dot(
            weights['W2'], self.activations['A1']) + biases['b2']
        self.activations['A2'] = sig(self.activations['Z2'])

        return self.activations['A2']

    def back_propogation(self, y):
        """
        Performs backpropagation to calculate the gradients of biases and weights.

        Args:
            y (ndarray): The target output array.

        Returns:
            dict: The gradients of biases and weights for each network layer.
        """

        # Note that I have not utilised this value in evaluation
        def cost_function(a, y):
            """
            Calculates the quadratic cost for the given dataset.

            Args:
                a (ndarray): The predicted output array.
                y (ndarray): The target output array.

            Returns:
                float: The mean squared error between the prediction and target values.
            """
            return np.sum((a - y) ** 2) / 2

        weights = self.weights
        biases = self.biases

        # Output layer
        # Calculate error for the output layer first by multiplying the cost derivative by the activation derivative
        delta2 = np.multiply(
            self.activations['A2'] - y, sig_deriv(self.activations['Z2']))
        # Assign calculated error and gradients for the corresponding elements in the gradient lists
        dW2 = np.outer(delta2, self.activations['A1'])
        db2 = delta2

        # Hidden layer
        delta1 = np.dot(weights['W2'].T, delta2) * \
            sig_deriv(self.activations['Z1'])
        dW1 = np.outer(delta1, self.activations['A0'])
        db1 = delta1

        # Calculate quadratic cost
        cost = cost_function(self.activations['A2'], y)

        # Return corresponding dictionary values
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'cost': cost}

    def update_parameters(self, gradients):
        """
        Updates the biases and weights of the neural network.

        Args:
            gradients (dict): The gradients of biases and weights for each network layer.
        """
        weights = self.weights
        biases = self.biases

        # Loop through weights and biases for the hidden and output layer
        for param in ['W1', 'b1', 'W2', 'b2']:
            # For each, update weights and biases scaled by the learning rate and size of mini-batches

            # Check if the current parameter is a weight
            if param.startswith('W'):
                weights[param] -= (self.learning_rate /
                                   self.batch_size) * gradients['d' + param]

            # Else its a bias
            else:
                biases[param] -= (self.learning_rate /
                                  self.batch_size) * gradients['d' + param]

    def train(self, x_train, y_train, x_test, y_test):
        """
        Trains the Feed-Forward Neural Network using the training data.

        Args:
            x_train (ndarray): The training data inputs.
            y_train (ndarray): The training data labels.
            x_test (ndarray): The test data inputs.
            y_test (ndarray): The test data labels.

        Returns:
            tuple: A tuple containing the lists of accuracies and costs during training.
        """
        # Initialise empty lists of accuracies and costs
        accuracies = []
        costs = []

        # Loop through each epoch
        for epoch in range(self.epochs):
            # Shuffle training data
            perm = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[perm]
            y_train_shuffled = y_train[perm]

            # Split into mini-batches
            mini_batches = [
                (x_train_shuffled[k:k+self.batch_size],
                 y_train_shuffled[k:k+self.batch_size])
                for k in range(0, len(x_train_shuffled), self.batch_size)
            ]

            # Train on mini-batches
            for mini_batch in mini_batches:
                x_mini_batch, y_mini_batch = mini_batch

                # Initialise gradient values as arrays of zeros
                gradients_sum = {
                    'dW1': np.zeros(self.weights['W1'].shape),
                    'db1': np.zeros(self.biases['b1'].shape),
                    'dW2': np.zeros(self.weights['W2'].shape),
                    'db2': np.zeros(self.biases['b2'].shape)
                }

                # Compute gradients for current mini-batch
                for x, y in zip(x_mini_batch, y_mini_batch):
                    # Feed forward for the current input
                    self.feed_forward(x)
                    # Utilise back propogation to calculate gradients for current output
                    gradients = self.back_propogation(y)

                    # Collect gradients for the parameter update
                    for param in gradients_sum:
                        gradients_sum[param] += gradients[param]

                    # Collect costs
                    costs.append(gradients['cost'])

                # Update parameters
                self.update_parameters(gradients_sum)

            # Evaluate on test data
            accuracy = self.evaluate(x_test, y_test)
            accuracies.append(accuracy)

        return accuracies, costs

    def evaluate(self, x_test, y_test):
        """
        Evaluates the accuracy of the predicted labels for the given test data.

        Args:
            x_test (ndarray): The test data inputs.
            y_test (ndarray): The test data labels.

        Returns:
            float: The accuracy of the predicted labels.
        """
        # Initialise empty list of predictions and set counter for correct predictions to 0
        predictions = []
        correct = 0

        # Iterate over test data
        for x, y in zip(x_test, y_test):
            # Feed forward input data
            output = self.feed_forward(x)

            # Get predicted and true labels
            predicted_label = np.argmax(output)
            true_label = np.argmax(y)
            predictions.append((predicted_label, true_label))

            # Compare predicted labels against true labels
            if predicted_label == true_label:
                # Increment counter if correct prediction
                correct += 1

        # Calculate accuracy and return as a percentage of the total input
        accuracy = correct / len(x_test) * 100

        return accuracy

##############
### DRIVER ###
##############


def main():
    """
    Main function to train, build, test, and graph the output results of the Feed-Forward Neural Network.
    """
    print('=========================================================================')
    print("Loading and pre-processing data files...")

    # Load CIFAR-10 dataset
    train_data_dict = unpickle('data_batch_1')
    test_data_dict = unpickle('test_batch')

    # Preprocess data
    x_train, y_train = preprocess_data(train_data_dict)
    x_test, y_test = preprocess_data(test_data_dict)

    # Apply one hot encoding to labels
    y_train = np.array(
        [encode_labels(digit) for digit in y_train]
    )
    y_test = np.array(
        [encode_labels(digit) for digit in y_test]
    )

    ##############
    ### TASK 1 ###
    ##############

    # Task 1: Create the base model to find maximum accuracy
    print('=========================================================================')
    print(f"Task 1: Create a neural network of size {SIZES}")

    print("Training base model...")

    # Create and train neural network
    nn = FFNeuralNet(
        sizes=SIZES, learning_rate=DFT_LEARNING_RATE, batch_size=DFT_BATCH_SIZE, epochs=EPOCHS
    )
    accuracies, costs = nn.train(x_train, y_train, x_test, y_test)

    # Print maximum accuracy achieved
    max_accuracy = max(accuracies)
    print(f"Maximum accuracy achieved: {max_accuracy:.3f} %")

    # Plot test accuracy vs epoch
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies)

    # Set visuals
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Epoch')

    # Save .png of plot
    plt.savefig('task1accuracy.png')

    ##############
    ### TASK 2 ###
    ##############

    # Task 2: Compare learning rates
    # Initialise empty lists to store maximum accuracies and plot values
    task2_accuracies = []
    task2_plots = []

    print('=========================================================================')
    print(f"Task 2: Compare learning rates: {LEARNING_RATES}")

    # Loop through defined learning rates
    for learning_rate in LEARNING_RATES:
        print(f"Training model with learning rate: {learning_rate}")

        # Create and train a new neural network
        nn = FFNeuralNet(
            sizes=SIZES, learning_rate=learning_rate, batch_size=DFT_BATCH_SIZE, epochs=EPOCHS
        )
        accuracies, costs = nn.train(x_train, y_train, x_test, y_test)

        # Collect maximum accuracies and plot values
        task2_accuracies.append(max(accuracies))
        task2_plots.append(accuracies)

    # Print maximum accuracies for different learning rates
    for learning_rate, accuracy in zip(LEARNING_RATES, task2_accuracies):
        print(
            f"Learning rate: {learning_rate} - Maximum accuracy achieved: {accuracy:.3f} %")

    # Plot test accuracy vs epoch for different learning rates
    plt.figure(figsize=(10, 6))
    for learning_rate, accuracies in zip(LEARNING_RATES, task2_plots):
        plt.plot(
            range(1, len(accuracies) + 1), accuracies, label=f"Learning rate: {learning_rate}"
        )

    # Set visuals
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Epoch for Different Learning Rates')
    plt.legend()

    # Save .png of plot
    plt.savefig('task2accuracy.png')

    ##############
    ### TASK 3 ###
    ##############

    # Task 3: Compare mini-batch sizes
    # Initialise empty lists to store maximum accuracies, plot values and execution times
    task3_accuracies = []
    task3_plots = []
    task3_execution_times = []

    print('=========================================================================')
    print(f"Task 3: Compare mini-batch sizes: {MINI_BATCH_SIZES}")

    for batch_size in MINI_BATCH_SIZES:
        print(f"Training model with mini-batch size: {batch_size}")
        start_time = time.time()

        # Create and train a new neural network
        nn = FFNeuralNet(
            sizes=SIZES, learning_rate=DFT_LEARNING_RATE, batch_size=batch_size, epochs=EPOCHS
        )
        accuracies, costs = nn.train(x_train, y_train, x_test, y_test)

        # Collect maximum accuracies and plot values
        task3_accuracies.append(max(accuracies))
        task3_plots.append(accuracies)

        # Collect execution times
        execution_time = time.time() - start_time
        task3_execution_times.append(execution_time)

    # Print maximum accuracies and execution times for different mini-batch sizes
    for batch_size, accuracy, execution_time in zip(MINI_BATCH_SIZES,
                                                    task3_accuracies,
                                                    task3_execution_times):
        print(
            f"Mini-batch size: {batch_size} - Maximum accuracy achieved: {accuracy:.3f} %")
        print(f"Execution time: {execution_time} seconds")

    # Plot maximum test accuracy vs mini-batch size
    plt.figure(figsize=(10, 6))
    plt.plot(MINI_BATCH_SIZES, task3_accuracies, marker='o')

    # Set visuals
    plt.xlabel('Mini-batch Size')
    plt.ylabel('Maximum Test Accuracy')
    plt.title('Maximum Test Accuracy vs Mini-batch Size')

    # Save .png of plot
    plt.savefig('task3accuracy.png')

    print("\nProgram execution complete, please see plotted graph images saved in the program folder.\n")


# Execute driver function
if __name__ == '__main__':
    main()
