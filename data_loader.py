import numpy as np


def load_data(filepath: str) -> list[tuple[np.ndarray, int]]:
    data, lang_to_int, int_to_lang = get_data(filepath)
    num_languages = len(lang_to_int)

    training_data = [(x, lang_one_hot(y, num_languages)) for x, y in data]

    return training_data, lang_to_int, int_to_lang


def lang_one_hot(j, size):
    array = np.zeros((size, 1))
    array[j] = 1.0
    return array


def get_data(filepath: str) -> list[tuple[np.ndarray, int]]:
    sentences = []
    languages = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            sentence, language = line.strip().split("\t")
            sentences.append((sentence, language))
            languages.add(language)

    lang_to_int = lang_to_int_mapping(languages)
    int_to_lang= int_to_lang_mapping(languages)
    encoded_sentences, char_to_int = encode_sentences(sentences, lang_to_int)
    return encoded_sentences, lang_to_int, int_to_lang


def char_to_int_mapping(vocabulary):
    char_to_int = {char: index for index, char in enumerate(vocabulary)}
    return char_to_int


def lang_to_int_mapping(languages):
    return {language: index for index, language in enumerate(languages)}

def int_to_lang_mapping(languages):
    return {index: language for index, language in enumerate(languages)}

def encode_sentence(
    sentence: str, char_to_int: dict, vocab_size: int
) -> np.ndarray:
    """Encodes sentence into a one-hot matrix."""
    encoded_sentence = np.zeros((len(sentence), vocab_size))
    for i, char in enumerate(sentence):
        index = char_to_int[char]
        encoded_sentence[i, index] = 1
    return encoded_sentence


def padding_sentences(encoded_sentences, max_length, vocab_size):
    sentences_count = len(encoded_sentences)
    padded_sentences = np.zeros((sentences_count, max_length, vocab_size))
    padding_sentences_with_language = []
    for i, (sentence, language) in enumerate(encoded_sentences):
        length = sentence.shape[0]
        padded_sentences[i, :length, :] = sentence
        padding_sentences_with_language.append((padded_sentences[i], language))
    return padding_sentences_with_language


def encode_sentences(sentences: list[tuple], lang_to_int: dict):
    """Encodes sentences into padded one-hot character vectors"""
    vocabulary = set("".join(s for s, _ in sentences))

    char_to_int = char_to_int_mapping(vocabulary)
    vocab_size = len(vocabulary)

    encoded_sentences = []
    for sentence, language in sentences:
        encoded_lang = lang_to_int[language]
        encoded_sentence = encode_sentence(
            sentence, char_to_int, vocab_size,
        )
        encoded_sentences.append(
            (
                encoded_sentence, encoded_lang
            )
        )

    max_length_sentence = max(len(s) for s, _ in sentences)

    padded_sentences = padding_sentences(
        encoded_sentences, max_length_sentence, vocab_size
    )

    return padded_sentences, char_to_int
