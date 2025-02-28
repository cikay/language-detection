To run the code follow the steps below

Create and activate shell by pipenv package manager
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

Paste the following code to shell and it will run. It will print out the test prediction rate like `Epoch 0: 6789 / 12000` for each epoch
and lastly for given Kurdish Kurmaji sentence it will print out `Kurdish Kurmanji`

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

After completion you can test any sentence just by

```py
sentence = "the sentence you want"
lang = lang_detector.detect(sentence)

print(lang)
```