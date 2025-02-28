## Neural Network Language Detector

This project implements a neural network model for language detection across multiple languages. The system is trained on a comprehensive dataset containing 60,000 sentences spanning 10 different languages.

To run the code follow the steps below

Create and activate virtual environment using pipenv
```
pipenv shell 
```

Install dependencies
```
pipenv install
```

Open Python shell

```
python
```

Run the following code to train the model and test a sample sentence. It will print out the test prediction rate like `Epoch 0: 6789 / 12000` for each epoch and lastly for given Kurdish Kurmaji sentence it will print out `Kurdish Kurmanji`. Note that it will takes couples of time to complete

```py
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

```

To test additional sentences

```py
sentence = "the sentence you want"
lang = lang_detector.detect(sentence)

print(lang)
```