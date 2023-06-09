from keras.utils import pad_sequences, to_categorical
import numpy as np
import pandas as pd

from nerc.functions import margin, flatting, string2num
from nerc.data import Data
from nerc.word2vec import Model_Word2Vec

class Vectorization:
    # path = "E:\\word2vec\\gensim-data\\word2vec-google-news-300\\GoogleNews-vectors-negative300.bin"
    # word2vec_model = KeyedVectors.load_word2vec_format(path, binary=True)

    def __init__(self, data: Data, word2vec_model: Model_Word2Vec):
        self.data = data
        self.word2vec_model = word2vec_model

    def word2vec(self):
        Sentences = []
        for sentence in self.data.sentences_num:
            Sentence = []
            for word in sentence:
                try:
                    # Sentence.append(self.word2vec_model.get_vector(word))
                    Sentence.append(self.word2vec_model.wv(word))
                except Exception as e:
                    Sentence.append(self.word2vec_model.wv(word))
            Sentences.append(Sentence)
        self.data.sentences_num = Sentences

    def vectorized_x(self):
        self.word2vec()
        self.data.sentences_num = margin(
            self.data.sentences_num, batch_size=Data.PADDING_SIZE
        )
        self.data.sentences_num = flatting(self.data.sentences_num)
        self.data.x = np.array(self.data.sentences_num, dtype="float32")
    
    def tag2num(self):
        self.data.ner_tags_num = [
            [Data.unique_ner_tags.get(tag) for tag in tags]
            for tags in self.data.ner_tags_num
        ]

    def num2oneHotEncoding(self):
        NUM_CLASSES = 9
        self.data.ner_tags_num = [
            to_categorical(tags, num_classes=NUM_CLASSES)
            for tags in self.data.ner_tags_num
        ]

    def vectorized_y(self):
        self.tag2num()
        self.num2oneHotEncoding()
        self.data.y = np.array(flatting(self.data.ner_tags_num), dtype="float32")

    def __scaled(self, df: pd.DataFrame):
        # copy the data
        df_min_max_scaled = df.copy()
        # apply normalization techniques
        for column in df_min_max_scaled.columns:
            df_min_max_scaled[column] = (
                df_min_max_scaled[column] - df_min_max_scaled[column].min()
            ) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
        return df_min_max_scaled

    def vectorized_features(self):
        chunks = string2num(flatting(self.data.chunk_tags), Data.unique_chunk_tags)
        poss = string2num(flatting(self.data.pos_tags), Data.unique_pos_tags)
        features = flatting(self.data.features)
        df_tags = pd.DataFrame({"chunk_tags": chunks, "pos_tags": poss})
        df_features = pd.DataFrame(
            data=features,
            columns=["is_capitalize", "isupper", "islower", "istitle", "isdigit"],
        )
        df = pd.concat((df_tags, df_features), axis=1)
        self.data.features = self.__scaled(df).to_numpy(dtype="float32")

    def vectorized_positions(self):
        self.data.positions = flatting(self.data.positions)
        [self.data.positions[i].append(i) for i in range(len(self.data.positions))]
        self.data.positions = np.array(self.data.positions, dtype="float32")