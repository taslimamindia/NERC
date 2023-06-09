from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nerc.functions import zip_2D, unzip_2D
from nerc.data import Data



class Preprocessing:
    def __init__(self, data: Data, text=None, lang="english"):
        self.data = data
        self.text = text
        self.lang = lang

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
        punctuations = stopwords.words(self.lang) + punctuation
        sentences = zip_2D(
            self.data.sentences_num,
            self.data.ner_tags_num,
            self.data.chunk_tags,
            self.data.pos_tags,
            self.data.features,
            self.data.positions
        )
        sentences = [
            [
                triple
                for triple in sentence
                if triple[0] not in punctuations or triple[1] != "O"
            ]
            for sentence in sentences
        ]
        (
            self.data.sentences_num,
            self.data.ner_tags_num,
            self.data.chunk_tags,
            self.data.pos_tags,
            self.data.features,
            self.data.positions
        ) = unzip_2D(sentences)
