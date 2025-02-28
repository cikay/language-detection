def load_data(filepath):
    sentences = []
    languages = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            sentence, language = line.strip().split("\t")
            sentences.append((sentence, language))
            languages.add(language)

    return sentences, languages
