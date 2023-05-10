from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D

import tensorflow as tf
import numpy as np

from utils.data import Data
from utils.loading import Loading
from utils.preprocessing import Preprocessing
from utils.vectorization import Vectorization

# Hyperparameters
NUM_FILTERS = 256
KERNEL_SIZE = 3
DROPOUT_RATE = 0.5
BATCH_SIZE = 32
EPOCHS = 10


# Functions utils
## LSTM


def preprocess(data: Data):
    preprocessing = Preprocessing(data=data)
    preprocessing.lowercasing()
    preprocessing.lemmatize()
    # preprocessing.remove_stopword()
    data.unicity()


def vectorize(data: Data):
    # treat sentences
    vector = Vectorization(data=data)
    vector.vectorized_x()
    vector.vectorized_y()


def vectorize1D(data: Data):
    # treat words
    vector = Vectorization(data=data)
    vector.word2vec()
    vector.tag2num()
    Sentences, Tags = [], []
    [[Sentences.append(word) for word in sentence] for sentence in data.sentences_num]
    [[Tags.append(tag) for tag in tags] for tags in data.ner_tags_num]
    data.x, data.y = np.array(Sentences, dtype="float32"), np.array(
        Tags, dtype="float32"
    )


def checkDataset(train, test, valid):
    print("X_train", train.x.shape)
    print("y_train", train.y.shape, "\n")
    print("X_test", test.x.shape)
    print("y_test", test.y.shape, "\n")
    print("X_valid", valid.x.shape)
    print("y_valid", valid.y.shape)


def main():
    train = Loading("train.txt").data
    test = Loading("test.txt").data
    valid = Loading("valid.txt").data
    preprocess(train)
    preprocess(test)
    preprocess(valid)
    vectorize(train)
    vectorize(test)
    vectorize(valid)
    checkDataset(train, test, valid)
    return train, test, valid


def evaluation(test: Data, y_predict):
    true, false, total, predict = 0, 0, 0, 0
    x, y, z = test.y.shape
    for i in range(x):
        for j in range(y):
            real_tag = np.argmax(test.y[i][j])
            predict_tag = np.argmax(y_predict[i][j])
            if predict_tag == 0:
                predict += 1
            if real_tag != 0:
                total = total + 1
                if real_tag == predict_tag:
                    true = true + 1
                else:
                    false = false + 1
    print("----------------------- Evaluation -------------------------")
    print(test.y.shape)
    print(predict, x * y)
    print(
        true, false, total, round(true / total, 3), round(false / total, 3), end="\n\n"
    )


def evaluation1D(test: Data, y_predict):
    true, false, total, predict = 0, 0, 0, 0
    x, y = test.y.shape
    for i in range(x):
        real_tag = np.argmax(test.y[i])
        predict_tag = np.argmax(y_predict[i])
        if predict_tag == 0:
            predict += 1
        if real_tag != 0:
            total = total + 1
            if real_tag == predict_tag:
                true = true + 1
            else:
                false = false + 1
    print("----------------------- Evaluation -------------------------")
    print(test.y.shape)
    print(predict, x * y)
    print(
        true, false, total, round(true / total, 3), round(false / total, 3), end="\n\n"
    )


# Model LSTM
class Model_LSTM:
    def __init__(self):
        self.model_LSTM = Sequential()

    def architecture(self):
        # Define the model architecture
        self.model_LSTM.add(
            LSTM(
                256,
                input_shape=(Data.MAX_LENGTH, Data.VOCAB_SIZE),
                return_sequences=True,
                dropout=DROPOUT_RATE,
            )
        )
        self.model_LSTM.add(LSTM(128, return_sequences=True, dropout=DROPOUT_RATE))
        self.model_LSTM.add(LSTM(64, return_sequences=True, dropout=DROPOUT_RATE))
        self.model_LSTM.add(LSTM(32, return_sequences=True, dropout=DROPOUT_RATE))
        self.model_LSTM.add(Dense(len(Data.unique_ner_tags), activation="softmax"))

    def architecture1D(self):
        # Define the model architecture
        self.model_LSTM = Sequential()
        self.model_LSTM.add(
            LSTM(
                128,
                input_shape=(1, Data.VOCAB_SIZE),
                return_sequences=True,
                dropout=DROPOUT_RATE,
            )
        )
        self.model_LSTM.add(LSTM(64, return_sequences=True, dropout=DROPOUT_RATE))
        self.model_LSTM.add(LSTM(32, return_sequences=True, dropout=DROPOUT_RATE))
        self.model_LSTM.add(Dense(len(Data.unique_ner_tags), activation="softmax"))

    def summary(self):
        self.model_LSTM.summary()

    def trainning(self, train: Data, valid: Data = None):
        self.model_LSTM.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        if valid == None:
            self.model_LSTM.fit(train.x, train.y, batch_size=BATCH_SIZE, epochs=EPOCHS)
        else:
            self.model_LSTM.fit(
                train.x,
                train.y,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(valid.x, valid.y),
            )

    def testing(self, test: Data):
        return self.model_LSTM.evaluate(test.x, test.y)

    def predicting(self, test: Data):
        return self.model_LSTM.predict(test.x, batch_size=BATCH_SIZE)


def main_lstm(max_lengths: list = [50]):
    for max_length in max_lengths:
        Data.MAX_LENGTH = max_length
        train, test, valid = main()
        model_lstm = Model_LSTM()
        model_lstm.architecture()
        # model_lstm.architecture1D()
        model_lstm.summary()
        model_lstm.trainning(train, valid)
        model_lstm.testing(test)
        y_predict_lstm = model_lstm.predicting(test)
        evaluation(test, y_predict_lstm)


# Model CNN
class Model_CNN:
    def __init__(self):
        # Define the model architecture
        self.model = Sequential()
        self.model.add(
            Conv1D(
                64,
                KERNEL_SIZE,
                activation="relu",
                input_shape=(Data.MAX_LENGTH, Data.VOCAB_SIZE),
                padding="same",
            )
        )
        self.model.add(Dropout(DROPOUT_RATE))
        self.model.add(Conv1D(32, KERNEL_SIZE, activation="relu", padding="same"))
        self.model.add(Dropout(DROPOUT_RATE))
        self.model.add(Dense(len(Data.unique_ner_tags), activation="softmax"))

    def summary(self):
        self.model.summary()

    def trainning(self, train: Data, valid: Data = None):        
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        if valid == None:
            self.model.fit(train.x, train.y, batch_size=BATCH_SIZE, epochs=EPOCHS)
        else:
            self.model.fit(
                train.x,
                train.y,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(valid.x, valid.y),
            )

    def testing(self, test: Data):
        return self.model.evaluate(test.x, test.y)

    def predicting(self, test: Data):
        return self.model.predict(test.x, batch_size=BATCH_SIZE)


def main_cnn(max_lengths: list = [50]):
    for max_length in max_lengths:
        Data.MAX_LENGTH = max_length
        train, test, valid = main()
        model_cnn = Model_CNN()
        model_cnn.trainning(train, valid)
        model_cnn.testing(test)
        y_predict_cnn = model_cnn.predicting(test)
        evaluation(test, y_predict_cnn)
