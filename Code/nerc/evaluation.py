from nerc.preprocessing import Preprocessing
from nerc.vectorization import Vectorization
from nerc.word2vec import Model_Word2Vec
from nerc.tagChunker import NGramTagChunker
from nltk.corpus import conll2000
from numpy import argmax


class Evaluation:
    def __init__(self, model, text):
        self.model = model
        self.model.word2vec_model = None
        self.text = text
        self.ntc = NGramTagChunker(conll2000.chunked_sents())

    def preprocess(self):
        sentences = self.model.data.sentences.copy()
        if isinstance(self.text, str):
            process = Preprocessing(self.model.data, text=self.text)
            process.tokenize()
        else:
            self.model.data.sentences = self.text
        self.model.word2vec_model = Model_Word2Vec(self.model.data.sentences + sentences, self.model.data.VOCAB_SIZE)
        (
            self.model.data.sentences,
            self.model.data.pos_tags,
            self.model.data.chunk_tags
        ) = self.ntc.parse_sents(self.model.data.sentences)
        self.model.data.sentences_num = self.model.data.sentences
        self.model.data.ner_tags = [['O'] * len(sentence) for sentence in self.model.data.sentences]
        self.model.data.ner_tags_num = self.model.data.ner_tags
        self.model.data.get_positions()
        self.model.data.features_level()
        if isinstance(self.text, list):
            process = Preprocessing(self.model.data)
            process.remove_stopword()
        else:
            process.remove_stopword()
    
    def vectorize(self):
        if self.model.word2vec_model != None:
            vector = Vectorization(data=self.model.data, word2vec_model=self.model.word2vec_model)
            vector.vectorized_x()
            # vector.vectorized_y()
            vector.vectorized_features()
            vector.vectorized_positions()
        else: raise Exception("Error: Model Word2vec is not yet defined !!!")

    def predict(self):
        if self.model.status == 1:
            y_predict = self.model.model.predict(self.model.data.x)
        elif self.model.status == 2: 
            y_predict = self.model.model.predict([self.model.data.x, self.model.data.features])
        else: raise Exception("Error: Type de status is not yet defined !!!")
        text = self.model.data.sentences.copy()
        for k in range(len(y_predict)):
            i = int(self.model.data.positions[k][0])
            j = int(self.model.data.positions[k][1])
            text[i][j] = [text[i][j], self.model.data.idx2tag(argmax(y_predict[k]))]
        return text
