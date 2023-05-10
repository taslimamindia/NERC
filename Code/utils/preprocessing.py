from utils.data import Data

from nltk import word_tokenize, sent_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessing:
    def __init__(self, data: Data, text=None, lang="english"):
        self.data = data
        self.text = text
        self.lang = lang
        if text == None:
            self.data.sentences_num = self.data.sentences
            self.data.ner_tags_num = self.data.ner_tags

    def tokenize(self):
        if self.text != None:
            sentenses = [
                word_tokenize(sentence, language=self.lang)
                for sentence in sent_tokenize(self.text, language=self.lang)
            ]
            self.data.sentences = [
                [token for token in sentence if token not in stopwords.words(self.lang)]
                for sentence in sentenses
            ]
            self.data.sentences_num = self.data.sentences

    def lowercasing(self):
        self.data.sentences_num = [
            [word.lower() for word in sentence] for sentence in self.data.sentences_num
        ]

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        self.data.sentences_num = [
            [lemmatizer.lemmatize(word) for word in sentence]
            for sentence in self.data.sentences_num
        ]

    def remove_stopword(self):
        punctuation = [
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "{",
            "|",
            "}",
            "~",
        ]
        sentences = [
            [
                (self.data.sentences_num[i][j], self.data.ner_tags[i][j])
                for j in range(len(self.data.sentences_num[i]))
            ]
            for i in range(len(self.data.sentences_num))
        ]
        sentences = [
            [
                (token, tag)
                for token, tag in sentence
                if token not in stopwords.words(self.lang) + punctuation
            ]
            for sentence in sentences
        ]
        self.data.sentences_num = [
            [token for token, tag in sentence] for sentence in sentences
        ]
        self.data.ner_tags_num = [
            [tag for token, tag in sentence] for sentence in sentences
        ]
