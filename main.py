import numpy as np

from network import Network
from data_loader import load_data
from data_preparer import DataPreparer, CharEncoder


train_data_path = "./data/train.tsv"
sentences, languages = load_data(train_data_path)
encoder = CharEncoder(sentences)
data_praperer = DataPreparer(sentences, languages, encoder)

net = Network(
    [data_praperer.input_size, data_praperer.hidden_size, data_praperer.languages_count]
)

net.SGD(
    data_praperer.train_data,
    epochs=30,
    mini_batch_size=10,
    learning_rate=3.0,
    test_data=data_praperer.test_data,
)


def print_language(input):
    predicted_output = net.forward(input)
    highest_activation_neuron_index = np.argmax(predicted_output)
    return int_to_lang[highest_activation_neuron_index]
