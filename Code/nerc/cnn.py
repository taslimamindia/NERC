from keras import Model, Input
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Flatten,
    concatenate
)

from nerc.data import Data
from nerc.base import Base_Model
from nerc.word2vec import Model_Word2Vec


class Model_CNN(Base_Model):
    def __init__(self, data:Data, w2v:Model_Word2Vec):
        super().__init__(data=data, word2vec=w2v)

    def architecture(self, activation="sigmoid"):
        # Define the model architecture
        self.model = Sequential()
        self.model.add(
            Conv1D(
                128,
                self.data.KERNEL_SIZE,
                activation="relu",
                input_shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE),
                padding="same",
            )
        )
        self.model.add(Dropout(self.data.DROPOUT_RATE))
        self.model.add(Conv1D(64, self.data.KERNEL_SIZE, activation="relu", padding="same"))
        self.model.add(Dropout(self.data.DROPOUT_RATE))
        self.model.add(Conv1D(32, self.data.KERNEL_SIZE, activation="relu", padding="same"))
        self.model.add(Dropout(self.data.DROPOUT_RATE))
        self.model.add(Dense(9, activation=activation))  # len(self.data.unique_ner_tags)

    def architecture_1(self, activation="sigmoid"):
        # Define the model architecture
        self.model = Sequential()
        self.model.add(
            Conv1D(
                32,
                3,
                activation="relu",
                input_shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE),
            )
        )
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(16, 3, activation="relu"))
        self.model.add(MaxPooling1D(2))
        # Flatten layer
        self.model.add(Flatten())
        # Dense layers
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(9, activation=activation))

    def architecture_2(self, activation="sigmoid"):
        input1 = Input(
            shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE), name="input_1"
        )
        output1 = Conv1D(
            32, 3, activation="relu", input_shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE)
        )(input1)
        output1 = MaxPooling1D(2)(output1)
        output1 = Conv1D(16, 3, activation="relu")(output1)
        output1 = MaxPooling1D(2)(output1)
        # Flatten layer
        output1 = Flatten()(output1)
        # Dense layers
        output1 = Dense(128, activation="relu")(output1)
        output1 = Dense(64, activation="relu")(output1)
        output1 = Dense(32, activation="relu")(output1)
        output1 = Dense(9, activation=activation)(output1)
        self.model = Model(inputs=input1, outputs=output1)

    def architecture_word2vec(self, activation="sigmoid"):
        input1 = Input(
            shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE), name="input_1"
        )
        output1 = Conv1D(64, 3, padding="same", activation="relu")(input1)
        output1 = MaxPooling1D(2, padding="same")(output1)
        output1 = Dropout(self.data.DROPOUT_RATE)(output1)
        output1 = Conv1D(32, 3, padding="same", activation="relu")(output1)
        output1 = MaxPooling1D(2, padding="same")(output1)
        output1 = Dropout(self.data.DROPOUT_RATE)(output1)
        # Flatten layer
        output1 = Flatten()(output1)
        # Dense layers
        # output1 = Dense(128, activation='relu')(output1)
        output1 = Dense(64, activation="relu")(output1)
        output1 = Dense(32, activation="relu")(output1)
        output1 = Dense(9, activation=activation)(output1)
        self.model = Model(inputs=input1, outputs=output1)

    def architecture_features(self, activation="sigmoid"):
        input2 = Input(shape=self.data.features.shape[1:])
        output2 = Dense(7, activation='relu')(input2)
        output2 = Dense(10, activation="relu")(output2)
        output2 = Dense(len(self.data.unique_ner_tags), activation=activation)(output2)
        self.model =  Model(inputs=input2, outputs=output2)
        
    def architecture_word2vec_features(self, activation="sigmoid"):
        input1 = Input(
            shape=(self.data.PADDING_SIZE, self.data.VOCAB_SIZE), name="input_1"
        )
        output1 = Conv1D(64, 3, padding="same", activation="relu")(input1)
        output1 = MaxPooling1D(2, padding="same")(output1)
        output1 = Dropout(self.data.DROPOUT_RATE)(output1)
        output1 = Conv1D(32, 3, padding="same", activation="relu")(output1)
        output1 = MaxPooling1D(2, padding="same")(output1)
        output1 = Dropout(self.data.DROPOUT_RATE)(output1)
        # Flatten layer
        output1 = Flatten()(output1)
        # Dense layers
        # output1 = Dense(128, activation='relu')(output1)
        output1 = Dense(64, activation="relu")(output1)
        output1 = Dense(32, activation="relu")(output1)
        output1 = Dense(9, activation=activation)(output1)
        model1 = Model(inputs=input1, outputs=output1)

        # Fully connected for Features
        input2 = Input(shape=self.data.features.shape[1:], name="input_2")
        output2 = Dense(7, activation="relu")(input2)
        # output2 = Dense(9, activation="relu")(output2)
        model2 =  Model(inputs=input2, outputs=output2)

        # Output model for the concatenation of CNN for word2vec and Fully Connected
        model1_model2 = concatenate([model1.output, model2.output])
        output3 = Dense(16, activation="relu")(model1_model2)
        output3 = Dense(9, activation=activation)(output3)
        self.model = Model(inputs=[model1.input, model2.input], outputs=output3)
