import numpy as np

from data_preparer import DataPreparer
from network import Network


class LanguageDetector:
    def __init__(self, sentences, languages, encoder):
        self.data_praperer = DataPreparer(sentences, languages, encoder)
        self.network = Network(
            [
                self.data_praperer.input_size,
                self.data_praperer.hidden_size,
                self.data_praperer.languages_count
            ]
        )
        self.network.SGD(
            self.data_praperer.train_data,
            epochs=30,
            mini_batch_size=10,
            learning_rate=3.0,
            test_data=self.data_praperer.test_data,
        )

    def detect(self, sentence):
        sentence_as_array = self.data_praperer.sentence_as_array(sentence)
        output = self.network.forward(sentence_as_array)
        highest_activation_neuron_index = np.argmax(output)
        index_int = int(highest_activation_neuron_index)
        return self.data_praperer.int_to_lang[index_int]
