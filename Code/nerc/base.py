from sklearn.model_selection import train_test_split

from nerc.functions import evaluation, matching_array
from nerc.data import Data
from nerc.word2vec import Model_Word2Vec
from nerc.preprocessing import Preprocessing
from nerc.vectorization import Vectorization


class Base_Model:
    def __init__(self, data: Data, word2vec: Model_Word2Vec, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid
        self.data = data
        self.word2vec_model = word2vec
        self.model = None
        self.history = None

    def train_test_split(self):
        # Split the training, testing, and validation
        x_train, x_test, self.train.y, self.test.y = train_test_split(
            self.data.positions, self.data.y, test_size=0.2
        )
        # Train Set:
        self.train.x = matching_array(x_train, self.data.x)
        self.train.features = matching_array(x_train, self.data.features)
        # Test Set:
        self.test.x = matching_array(x_test, self.data.x)
        self.test.features = matching_array(x_test, self.data.features)
        
    def train_test_valid_split(self):
        # Split the training, testing, and validation
        x_train, x_test, self.train.y, self.test.y = train_test_split(
            self.data.positions, self.data.y, test_size=0.2
        )
        x_train, x_valid, self.train.y, self.valid.y = train_test_split(
            x_train, self.train.y, test_size=0.15
        )
        # Train Set:
        self.train.x = matching_array(x_train, self.data.x)
        self.train.features = matching_array(x_train, self.data.features)
        # Test Set:
        self.test.x = matching_array(x_test, self.data.x)
        self.test.features = matching_array(x_test, self.data.features)
        # valid Set:
        self.valid.x = matching_array(x_valid, self.data.x)
        self.valid.features = matching_array(x_valid, self.data.features)

    def change(self, max_length=None, vocab_size=None, padding_size=None):
        if max_length != None:
            self.data.MAX_LENGTH = max_length
        if vocab_size != None:
            self.data.VOCAB_SIZE = vocab_size
        if padding_size != None:
            self.data.PADDING_SIZE = padding_size

    def preprocessing(self):
        self.data.get_positions()
        self.data.features_level()
        preprocessing = Preprocessing(data=self.data)
        preprocessing.remove_stopword()

    def vectorization(self):
        vector = Vectorization(data=self.data, word2vec_model=self.word2vec_model)
        vector.vectorized_x()
        vector.vectorized_y()
        vector.vectorized_features()
        vector.vectorized_positions()

    def summary(self):
        self.model.summary()

    def training(
        self,
        x_train,
        y_train,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.data.BATCH_SIZE,
            epochs=self.data.EPOCHS,
        )

    def training_valid(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)    
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.data.BATCH_SIZE,
            epochs=self.data.EPOCHS,
            validation_data=(x_valid, y_valid),
        )

    def testing(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predicting(self, x_test, y_test):
        y_predict = self.model.predict(x_test, batch_size=self.data.BATCH_SIZE)
        evaluation(y_test, y_predict, self.data.unique_ner_tags.get("O"))
