import numpy as np
import random


def sigmoid(weighted_input: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-weighted_input))


def sigmoid_prime(weighted_input: np.ndarray) -> np.ndarray:
    return sigmoid(weighted_input) * (1 - sigmoid(weighted_input))


class Network:
    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # we starts from 1. index to skip the input layer
        # we use 1 as column size as biases are scalar values
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # we use y as the next layer size and x as the current layer size
        # Network([2, 3, 1])
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, input: np.ndarray) -> np.ndarray:
        current_output = input
        for bias, weight in zip(self.biases, self.weights):
            weighted_input = np.dot(weight, current_output) + bias
            current_output = sigmoid(weighted_input)

        return current_output

    def SGD(
        self,
        training_data: list[tuple[np.ndarray, np.ndarray]],
        epochs: int,
        mini_batch_size: int,
        learning_rate: float,
        test_data: list[tuple[np.ndarray, np.ndarray | int]] | None = None,
    ):
        if test_data:
            n_test = len(test_data)

        training_data_len = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, training_data_len, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(
        self,
        mini_batch: list[tuple[np.ndarray, np.ndarray]],
        learning_rate: float,
    ):
        nabla_biases, nabla_weights = self._compute_nable(mini_batch)

        new_weights = []
        for weight, nabla_weight in zip(self.weights, nabla_weights):
            weight -= (learning_rate / len(mini_batch)) * nabla_weight
            new_weights.append(weight)

        self.weights = new_weights

        new_biases = []
        for bias, nambla_bias in zip(self.biases, nabla_biases):
            bias -= (learning_rate / len(mini_batch)) * nambla_bias
            new_biases.append(bias)

        self.biases = new_biases

    def _backprop(
        self, input: np.ndarray, desired_output: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:

        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        layers_activation, layers_weighted_input = self.collect_layers_values(input)

        output_error = self.cost_derivative(layers_activation[-1], desired_output)
        delta = output_error * sigmoid_prime(layers_weighted_input[-1])

        nabla_biases[-1] = delta
        nabla_weights[-1] = np.dot(delta, layers_activation[-2].transpose())

        for layer in range(2, self.num_layers):
            weighted_input = layers_weighted_input[-layer]

            output_error = np.dot(self.weights[-layer + 1].transpose(), delta)
            delta = output_error * sigmoid_prime(weighted_input)

            nabla_biases[-layer] = delta
            nabla_weights[-layer] = np.dot(
                delta, layers_activation[-layer - 1].transpose()
            )

        return nabla_biases, nabla_weights

    def _compute_nable(
        self, mini_batch: list[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:

        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        for input, desired_output in mini_batch:
            delta_nabla_biases, delta_nabla_weights = self._backprop(
                input, desired_output
            )

            nabla_biases = [
                nabla_bias + delta_nabla_bias
                for nabla_bias, delta_nabla_bias in zip(
                    nabla_biases, delta_nabla_biases
                )
            ]
            nabla_weights = [
                nabla_weight + delta_nabla_weight
                for nabla_weight, delta_nabla_weight in zip(
                    nabla_weights, delta_nabla_weights
                )
            ]

        return nabla_biases, nabla_weights

    def evaluate(
        self, test_data: list[tuple[np.ndarray, np.ndarray | int]]
    ) -> int:
        test_results = []
        for input, output in test_data:
            predicted_output = self.forward(input)
            highest_activation_neuron_index = np.argmax(predicted_output)
            test_results.append((highest_activation_neuron_index, output))

        return sum(int(x == y) for x, y in test_results)

    def collect_layers_values(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activation = input
        layers_activation = [input]
        layers_weighted_input = []
        for bias, weight in zip(self.biases, self.weights):
            weighted_input = np.dot(weight, activation) + bias
            layers_weighted_input.append(weighted_input)

            activation = sigmoid(weighted_input)
            layers_activation.append(activation)

        return layers_activation, layers_weighted_input

    def cost_derivative(
        self, output_activations: np.ndarray, desired_output: np.ndarray
    ) -> np.ndarray:
        return 2 * (output_activations - desired_output)
