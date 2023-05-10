from utils.data import Data
import os

class Loading():
    def __init__(self, path):
        if os.path.exists("../Data/conll2003_english/"): 
            base_file = "../Data/conll2003_english/"
        else:
            base_file = "/content/NERC/Data/conll2003_english/"        
        self.data = Data()
        self.load_sentences(base_file + path)
    def load_sentences(self, filepath):
        tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                    if len(tokens) > 0:
                        self.data.sentences.append(tokens)
                        self.data.pos_tags.append(pos_tags)
                        self.data.chunk_tags.append(chunk_tags)
                        self.data.ner_tags.append(ner_tags)
                        tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
                else:
                    l = line.split(' ')
                    tokens.append(l[0])
                    pos_tags.append(l[1])
                    chunk_tags.append(l[2])
                    ner_tags.append(l[3].strip('\n'))