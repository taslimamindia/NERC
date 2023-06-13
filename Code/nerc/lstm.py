from nerc.base import Base_Model
from nerc.data import Data
from nerc.word2vec import Model_Word2Vec

from keras.models import Sequential
from keras.layers import LSTM, Dense, concatenate, Flatten
from keras import Model, Input

class Model_LSTM(Base_Model):
    def __init__(self, data:Data, w2v:Model_Word2Vec, train=None, test=None, valid=None):
        super().__init__(data=data, word2vec=w2v, train=train, test=test, valid=valid)

    def architecture_word2vec(self, activation="sigmoid"):
        input_lstm = Input(shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE))
        output_lstm = LSTM(256, return_sequences=True, dropout=self.data.DROPOUT_RATE)(input_lstm)
        output_lstm = LSTM(128, return_sequences=True, dropout=self.data.DROPOUT_RATE)(output_lstm)
        output_lstm = LSTM(64, return_sequences=True, dropout=self.data.DROPOUT_RATE)(output_lstm)
        output_lstm = LSTM(32, return_sequences=True, dropout=self.data.DROPOUT_RATE)(output_lstm)
        output_lstm = Flatten()(output_lstm)
        output_lstm = Dense(64, activation="relu")(output_lstm)
        output_lstm = Dense(32, activation="relu")(output_lstm)
        output_lstm = Dense(9, activation=activation)(output_lstm)
        self.model = Model(inputs=input_lstm, outputs=output_lstm)
    
    def architecture_word2vec_features(self, activation="sigmoid"):
        input1 = Input(shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE), name="input_1")
        output1 = LSTM(256, return_sequences=True, dropout=self.data.DROPOUT_RATE)(input1)
        output1 = LSTM(128, return_sequences=True, dropout=self.data.DROPOUT_RATE)(output1)
        output1 = LSTM(64, return_sequences=True, dropout=self.data.DROPOUT_RATE)(output1)
        output1 = LSTM(32, return_sequences=True, dropout=self.data.DROPOUT_RATE)(output1)
        output1 = Flatten()(output1)
        output1= Dense(64, activation="relu")(output1)
        output1= Dense(32, activation="relu")(output1)
        output1= Dense(9, activation="relu")(output1)
        model1 = Model(inputs=input1, outputs=output1)
        # Fully connected for Features
        input2 = Input(shape=self.data.features.shape[1:], name="input_2")
        output2 = Dense(7, activation="relu")(input2)
        model2 =  Model(inputs=input2, outputs=output2)
        # Output model for the concatenation of CNN for word2vec and Fully Connected
        model1_model2 = concatenate([model1.output, model2.output])
        output3 = Dense(16, activation="relu")(model1_model2)
        output3 = Dense(9, activation=activation)(output3)
        self.model = Model(inputs=[model1.input, model2.input], outputs=output3)