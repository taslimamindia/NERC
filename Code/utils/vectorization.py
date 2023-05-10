from utils.data import Data

import numpy as np
from keras.utils import to_categorical, pad_sequences

from gensim.models import Word2Vec


class Vectorization():
    def __init__(self, data:Data):
        self.data = data
    def word2vec(self, min_count=1, window=5):
        word2vec_model = Word2Vec(self.data.sentences_num, min_count=min_count, vector_size=Data.VOCAB_SIZE, window=window)
        self.data.sentences_num = [[word2vec_model.wv[word] for word in sentence] for sentence in self.data.sentences_num]
    def padding_x(self, value=np.zeros((Data.VOCAB_SIZE,), dtype="float32"), dtype="float32"):
        self.data.x = pad_sequences(
            sequences=self.data.sentences_num, 
            maxlen=self.data.MAX_LENGTH, 
            dtype=dtype, 
            padding="post", 
            value=value
        )
    def vectorized_x(self):
        self.word2vec()
        self.padding_x()
    def tag2num(self):
        NUM_CLASSES = len(Data.unique_ner_tags)
        self.data.ner_tags_num = [[to_categorical(Data.unique_ner_tags.get(tag), num_classes=NUM_CLASSES) for tag in tags] for tags in self.data.ner_tags_num]
    def padding_y(self, value=to_categorical(Data.unique_ner_tags.get("O"), num_classes=len(Data.unique_ner_tags))):
        self.data.y = pad_sequences(
            sequences=self.data.ner_tags_num, 
            maxlen=self.data.MAX_LENGTH,
            padding="post", 
            dtype="float32",
            value=value
        )
    def vectorized_y(self):
        self.tag2num()
        self.padding_y()