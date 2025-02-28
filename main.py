from data_loader import load_data
from data_preparer import CharEncoder
from language_detector import LanguageDetector


train_data_path = "./data/train.tsv"
sentences, languages = load_data(train_data_path)
encoder = CharEncoder(sentences)
lang_detector = LanguageDetector(sentences, languages, encoder)
sentence = "Ez ê di vê gotarê da qala ên ku ez guhdar û temaşe dikim bikim."
lang = lang_detector.detect(sentence)

print(lang)
