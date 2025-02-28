from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    def __init__(self, sentences):
        self.sentences = sentences

    @abstractmethod
    def encode_sentence(self, sentence: str) -> np.ndarray:
        pass


class CharEncoder:
    def __init__(self, sentences):
        self.vocabulary = set("".join(s for s, _ in sentences))
        self.vocabulary_size = len(self.vocabulary)
        self.char_to_int = self._char_to_int_mapping()

    def encode_sentence(self, sentence: str) -> np.ndarray:
        """Encodes sentence into a one-hot matrix."""
        encoded_sentence = np.zeros((len(sentence), self.vocabulary_size))
        for i, char in enumerate(sentence):
            index = self.char_to_int[char]
            encoded_sentence[i, index] = 1
        return encoded_sentence

    def _char_to_int_mapping(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.vocabulary)}


class DataPreparer:
    def __init__(self, sentences, languages: set[str], encoder: Encoder):
        self.sentences = sentences
        self.languages = languages
        self.languages_count = len(languages)
        self.max_length_sentence = max(len(s) for s, _ in self.sentences)
        self.vocabulary = set("".join(s for s, _ in self.sentences))
        self.vocabulary_size = len(self.vocabulary)
        self.lang_to_int = self._lang_to_int_mapping()
        self.int_to_lang = self._int_to_lang_mapping()
        self.encoder = encoder
        data = self.preprocess_data()
        self.train_data = data[: int(len(data) * 0.8)]
        self.test_data = [
            (x.reshape(-1, 1), np.argmax(y)) for x, y in data[int(len(data) * 0.8) :]
        ]
        shape = self.train_data[0][0].shape
        self.input_size = shape[0] * shape[1]
        self.hidden_size = 10

    def preprocess_data(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Encodes sentences into padded one-hot character vectors"""
        encoded_sentences = []
        for sentence, language in self.sentences:
            encoded_lang = self.lang_to_int[language]
            lang_array = self.lang_one_hot(encoded_lang, self.languages_count)
            encoded_sentence = self.encoder.encode_sentence(sentence)
            encoded_sentences.append((encoded_sentence, lang_array))

        padded_sentences = self._padding_sentences(encoded_sentences)
        training_data = [(x.reshape(-1, 1), y) for x, y in padded_sentences]

        return training_data

    def sentence_as_array(self, sentence):
        encoded_sentence = self.encoder.encode_sentence(sentence)
        padded_sentence = self._padding_sentence(encoded_sentence)
        return padded_sentence.reshape(-1, 1)

    def _padding_sentences(
        self, encoded_sentences: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        sentences_count = len(encoded_sentences)
        padded_sentences = np.zeros(
            (sentences_count, self.max_length_sentence, self.vocabulary_size)
        )
        padding_sentences_with_language = []
        for i, (sentence, language) in enumerate(encoded_sentences):
            length = sentence.shape[0]
            padded_sentences[i, :length, :] = sentence
            padding_sentences_with_language.append((padded_sentences[i], language))

        return padding_sentences_with_language

    def _lang_to_int_mapping(self) -> dict[str, int]:
        return {language: index for index, language in enumerate(self.languages)}

    def _int_to_lang_mapping(self) -> dict[int, str]:
        return {index: language for index, language in enumerate(self.languages)}

    def _padding_sentence(self, encoded_sentence):
        padded_sentence = np.zeros((1, self.max_length_sentence, self.vocabulary_size))
        length = encoded_sentence.shape[0]
        padded_sentence[0, :length, :] = encoded_sentence
        return padded_sentence

    @staticmethod
    def lang_one_hot(j, size):
        array = np.zeros((size, 1))
        array[j] = 1.0
        return array
