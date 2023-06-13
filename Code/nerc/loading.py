import os
from nerc.data import Data


class Loading:
    def __init__(self, data:Data, path, names):
        self.data = data
        for name in names:
            self.__load_sentences(path + name)

    def __load_sentences(self, filepath):
        tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
        with open(filepath, "r") as f:
            for line in f.readlines():
                if line == ("-DOCSTART- -X- -X- O\n") or line == "\n":
                    if len(tokens) > 0:
                        self.data.sentences.append(tokens)
                        self.data.pos_tags.append(pos_tags)
                        self.data.chunk_tags.append(chunk_tags)
                        self.data.ner_tags.append(ner_tags)
                        tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
                else:
                    l = line.split(" ")
                    tokens.append(l[0])
                    pos_tags.append(l[1])
                    chunk_tags.append(l[2])
                    ner_tags.append(l[3].strip("\n"))
            self.data.sentences_num = self.data.sentences
            self.data.ner_tags_num = self.data.ner_tags
