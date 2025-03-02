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

        # Each item represents a layer but first layer as it is input and has no biases
        # we use 1 as column size as biases are scalar values
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # creates connects starts from first layer through hidden layers to output layer
        # When layers number is n, n - 1 sets of weights connecting them.
        # Start from first to second until last-to-second to last
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

        layers_activation, layers_weighted_input = self.compute_activations_and_weighted_inputs(input)

        output_error = self.cost_derivative(layers_activation[-1], desired_output)
        delta = output_error * sigmoid_prime(layers_weighted_input[-1])

        nabla_biases[-1] = delta
        nabla_weights[-1] = np.dot(delta, layers_activation[-2].transpose())

        # Iterate backwards from second to last layer towards the first hidden layer.
        # First layer in not included since it is input no biases and weights to adjust
        for layer in range(2, self.num_layers):
            current_layer_index = -layer
            next_layer_index = -layer + 1
            previous_layer_index = -layer - 1

            weighted_input = layers_weighted_input[current_layer_index]

            # weights between current and next layer
            current_to_next_layer_weights = self.weights[next_layer_index]
            next_to_current_layer_weights = current_to_next_layer_weights.transpose()

            output_error = np.dot(next_to_current_layer_weights, delta)
            # delta is the error signal from the next layer
            delta = output_error * sigmoid_prime(weighted_input)

            nabla_biases[current_layer_index] = delta
            nabla_weights[current_layer_index] = np.dot(
                delta, layers_activation[previous_layer_index].transpose()
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

    def compute_activations_and_weighted_inputs(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Based on input, compute the current activations and weighted inputs for each layer during the forward pass.
        These intermediate values will be used to determine the error of our biases and weights for this input.
        """

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
