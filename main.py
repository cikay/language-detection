import numpy as np

from network import Network
from data_loader import load_data

train_data_path = "./data/train.tsv"
data, lang_to_int, int_to_lang = load_data(train_data_path)


print("Data loaded", len(data))
train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]
input_shape = train_data[0][0].shape
print("Input shape:", input_shape)
input_size = (
    input_shape[0] * input_shape[1]
)
print("Flattened input size:", input_size)
num_languages = len(lang_to_int)
hidden_size = 10
print("input size: ", input_size)
net = Network([input_size, hidden_size, num_languages])
flattened_train_data = [(x.reshape(-1, 1), y) for x, y in train_data]
flattened_test_data = [(x.reshape(-1, 1), np.argmax(y)) for x, y in test_data]
net.SGD(
    flattened_train_data,
    epochs=30,
    mini_batch_size=10,
    learning_rate=3.0,
    test_data=flattened_test_data,
)


def print_language(input):
    predicted_output = net.forward(input)
    highest_activation_neuron_index = np.argmax(predicted_output)
    return int_to_lang[highest_activation_neuron_index]
