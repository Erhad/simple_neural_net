import random

import numpy as np
import _pickle


class Network:
    def __init__(self, shape):

        """Initializes weight and bias matrices using random floats from normal standard distribution.
        First layer does not have it's own weights and biases, as it is the input layer"""
        self.layers = len(shape)
        self.shape = shape
        self.biases = [np.random.randn(height, 1) for height in shape[1:]]
        self.weights = [np.random.randn(height, prevHeight) for height, prevHeight in zip(shape[1:], shape[:-1])]

    def backprop_get_gradient(self, input_matrix, correct_output_matrix):
        """Calculates errors of each layers weights and biases using backpropagation.
        Dot product of inputvise bias gradients and matrix2vec results in the vector that is the sum of
        those gradients"""

        delta_w = [np.zeros(np.shape(w)) for w in self.weights]
        delta_b = [np.zeros(np.shape(b)) for b in self.biases]
        matrix2vec = np.ones((input_matrix.shape[1], 1))

        """Get zs and activations of the network with a given input and calculate the derivative of the cost function"""
        zs, activations = self.zs_and_activations(input_matrix)
        cost_derivative = self.cost_derivative(activations[-1], correct_output_matrix)

        """Calculate the error of the first layer, later backpropagate to calculate errors of the previous layers.
        Note the negative indexes used"""
        delta = np.multiply(cost_derivative, logistic_derivative(zs[-1]))
        delta_b[-1] = np.dot(delta, matrix2vec)
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.layers):
            wt_dot_delta = np.dot(self.weights[1 - layer].transpose(), delta)
            delta = np.multiply(wt_dot_delta, logistic_derivative(zs[-layer]))
            delta_b[-layer] = np.dot(delta, matrix2vec)
            delta_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return delta_b, delta_w


    def zs_and_activations(self, input_matrix):
        """Calculates and returns activations and z's of every layer.
        As our program can deal with multiple inputs at the same time,
        it needs to convert the bias vector to a bias matrix.
        Dot product of bias vector and vector_to_matrix
        does exactly that. All of the columns of the resultant
        matrix shall have the same entries as the b-vector"""
        zs = []
        activation = input_matrix
        activations = [activation]
        vec2matrix = np.ones((1, input_matrix.shape[1]))

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + np.dot(b, vec2matrix)
            zs.append(z)
            activation = logistic(z)
            activations.append(activation)
        return zs, activations

    def cost_derivative(self, network_output, desired_output):

        return network_output - desired_output

    def evaluate(self, test_data):
        """Return the number of correct outputs"""
        test_results = [(np.argmax(self.get_result(x)), y)
                        for (x, y) in zip(test_data[0], test_data[1])]
        return sum(int(x == y) for (x, y) in test_results)

    def get_result(self, a):
        """Returns the output of network when input=a"""
        for b, w in zip(self.biases, self.weights):
            a = logistic(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent. Training data is a tuple (x,y) with inputs and correct outputs
        respectively. When shuffling, it is converted to list, replacing (x,y)
         with data[0] and data[1] respectively"""
        if test_data:
            test_length = len(test_data[0])
        training_data_length = len(training_data[1])

        for i in range(epochs):

            random.shuffle(list(training_data))
            """Create mini batches"""
            for j in range(0, training_data_length, mini_batch_size):
                mini_batch_inputs = np.concatenate(training_data[0][j:j + mini_batch_size], 1)
                mini_batch_outputs = np.concatenate(training_data[1][j:j + mini_batch_size], 1)
                self.update_with_SGD(mini_batch_inputs, mini_batch_outputs, eta)

            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {test_length}")
            else:
                print(f"Epoch {i} complete")



    def evalNumber(self, inputVector):
        """Returns the number of the output node that got the highest activation"""
        return np.argmax(self.get_result(inputVector))


    def update_with_SGD(self, mini_batch_inputs, mini_batch_outputs, learning_rate):
        """Update weights and biases using stochastic gradient descent with batches."""

        delta_b, delta_w = self.backprop_get_gradient(mini_batch_inputs, mini_batch_outputs)
        length = len(mini_batch_inputs[0])

        self.weights = [w - (learning_rate / length) * w_gradient for w, w_gradient in zip(self.weights, delta_w)]
        self.biases = [b - (learning_rate / length) * b_gradient for b, b_gradient in zip(self.biases, delta_b)]

    def save(self):
        file_name = input("How do you want to name your file\n(dont include filetype)")
        file = open(fileName+".pickle", "wb")
        _pickle.dump(self, file)
        file.close()


def logistic(arr):
    return 1.0 / (1.0 + np.exp(-arr))


def logistic_derivative(arr):
    return logistic(arr) * (1 - logistic(arr))

