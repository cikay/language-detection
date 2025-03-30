## Neural Network Language Detector

A language detection model using neural networks to identify 10 different languages from text input. Built as a learning project with a dataset of 60,000 sentences, thanks to Mozilla Common Voice achieving 90% accuracy. This implementation demonstrates the fundamentals of neural network and text classification.

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

Run the following code to train the model and test a sample sentence. It will print out the test prediction rate like `Epoch 0: 6789 / 12000` for each epoch and lastly for given Kurdish Kurmaji sentence it will print out `Kurdish Kurmanji`. Note that it takes about 1 minute to print out first epoch results and 20 minutes for completion

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

print(lang) # Should print "Kurdish Kurmanji"

```

To test additional sentences

```py
sentence = "the sentence you want"
lang = lang_detector.detect(sentence)

print(lang)
```
