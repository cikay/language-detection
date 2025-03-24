import numpy as np
from collections import defaultdict


def sigmoid(x):
    x = np.clip(x, -100, 100)  # Avoid overflow
    return 1 / (1 + np.exp(-x))


class Word2Vec:
    def __init__(
        self, sentences, window_size=2, N=10, alpha=0.0001, negative_samples=5
    ):
        self.sentences = sentences
        self.window_size = window_size
        self.N = N  # Embedding size
        self.alpha = alpha
        self.negative_samples = negative_samples
        self.unigram_table = []
        self.data = []
        self._build_vocab()
        self._build_unigram_table()
        self._generate_words_context()
        self._initialize_weights()

    def train(self, epochs):
        """Train the model using Skip-Gram with Negative Sampling."""
        initial_learning_rate = self.alpha
        for epoch in range(epochs):
            self.alpha = initial_learning_rate * (1 - epoch / epochs)  # Linear decay
            total_loss = 0
            for center_word_idx, context_word_idx in self.data:
                # Rows of W represent the vector of each center word
                center_word_vector = self.W[center_word_idx]
                # Columns of W1 represent the vector of each context word
                context_word_vector = self.W1[:, context_word_idx]
                # The dot product measures the similarity between the two vectors
                positive_score = np.dot(context_word_vector, center_word_vector)
                # The sigmoid function converts the score into a probability.
                positive_prob = sigmoid(positive_score)

                total_loss += -np.log(positive_prob)

                # Derivative of the loss function with respect to context word vector
                gradient_context_positive = (positive_prob - 1) * center_word_vector
                # By subtracting the gradient, we minimize the loss function
                self.W1[:, context_word_idx] -= gradient_context_positive * self.alpha
                gradient_center_positive = gradient_context_positive * self.W1[:, context_word_idx]

                # Negative samples
                gradient_center_negative = np.zeros_like(center_word_vector)
                for negative_word_idx in self._negative_sample(context_word_idx):
                    negative_vector = self.W1[:, negative_word_idx]
                    negative_score = np.dot(negative_vector, center_word_vector)
                    negative_prob = sigmoid(negative_score)
                    total_loss += -np.log(1 - negative_prob)

                    gradient_negative = self.alpha * negative_prob
                    self.W1[:, negative_word_idx] -= gradient_negative * center_word_vector
                    gradient_center_negative += (
                        gradient_negative * self.W1[:, negative_word_idx]
                    )

                # By subtracting the gradient, we minimize the loss function
                self.W[center_word_idx] -= (
                    gradient_center_positive + gradient_center_negative
                )

            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, word, top_n=3):
        """Find closest words to the given word."""
        if word not in self.word_to_index:
            print("Word not in vocabulary")
            return []

        idx = self.word_to_index[word]
        word_vector = self.W[idx] / np.linalg.norm(self.W[idx])  # Normalize
        similarities = np.dot(self.W, word_vector) / np.linalg.norm(self.W, axis=1)
        closest_indices = np.argsort(-similarities)[1 : top_n + 1]
        return [self.index_to_word[i] for i in closest_indices]

    def _build_vocab(self):
        """Build vocabulary and mappings."""
        self.index_to_word = list(
            set(word for sentence in self.sentences for word in sentence)
        )
        self.word_to_index = {word: i for i, word in enumerate(self.index_to_word)}
        self.V = len(self.index_to_word)

    def _build_unigram_table(self):
        """Build a unigram table for negative sampling."""
        word_counts = defaultdict(int)
        for sentence in self.sentences:
            for word in sentence:
                word_counts[word] += 1

        for word, count in word_counts.items():
            self.unigram_table.extend([self.word_to_index[word]] * int(count**0.75))

        self.unigram_table = np.array(self.unigram_table)

    def _generate_words_context(self):
        """Generate (center, context) word pairs."""
        for sentence in self.sentences:
            words_index = [self.word_to_index[word] for word in sentence]
            for i, center_word_idx in enumerate(words_index):
                from_ = max(0, i - self.window_size)
                to = min(i + self.window_size + 1, len(words_index))
                for j in range(from_, to):
                    if i == j:
                        continue

                    context_word_idx = words_index[j]
                    self.data.append((center_word_idx, context_word_idx))

    def _initialize_weights(self):
        # Each row represents the vector of a center word
        self.W = np.random.uniform(-0.5, 0.5, (self.V, self.N))
        # Each column represents the vector of a context word
        self.W1 = np.random.uniform(-0.5, 0.5, (self.N, self.V))

    def _negative_sample(self, context_idx):
        """Generate negative samples using the unigram table."""
        negatives = []
        while len(negatives) < self.negative_samples:
            neg = np.random.choice(self.unigram_table)
            if neg != context_idx:
                negatives.append(neg)
        return negatives


def prepare_data():
    with open("./data/train.tsv", "r") as file:
        corpus = []
        for line in file:
            line = line.strip().split("\t")
            lang = line[1]
            if lang == "Kurdish Kurmanji":
                sentence = line[0].lower().split()
                corpus.append(sentence)

        return corpus


corpus = prepare_data()

w2v = Word2Vec(corpus, N=200, window_size=5, negative_samples=10)
w2v.train(epochs=50)
print(w2v.predict("ez", 3))
