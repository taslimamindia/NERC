from nerc.data import Data

import numpy as np

# import unrar
# import pyunpack
from pickle import dump, load
from keras.utils import pad_sequences


# def unziprar(path_rar, dest_dir):
#     pyunpack.Archive(path_rar).extractall(dest_dir, auto_create_dir=True)


def reset_for_serialization(model):
    model.data.sentences_num = []
    model.data.ner_tags = []
    model.data.ner_tags_num = []
    model.data.chunk_tags = []
    model.data.pos_tags = []
    model.data.features = []
    model.data.positions = []
    model.data.x, model.data.y = None, None
    model.train, model.test, model.valid = None, None, None
    model.word2vec_model = None
    model.history = None


def matching_array(positions, data):
    return np.array(
        [data[int(pos3)] for pos1, pos2, pos3 in positions], dtype="float32"
    )


def serialization(data, path):
    with open(path, "wb") as outfile:
        dump(data, outfile)


def deserialization(path):
    with open(path, "rb") as infile:
        data = load(infile)
    return data


def load_data(data: Data, path, name):
    data.x = np.load(file=path + name + "_x.npy")
    data.features = np.load(file=path + name + "_features.npy")
    data.y = np.load(file=path + name + "_y.npy")


def save_data(data: Data, path, name):
    np.save(file=path + name + "_x.npy", arr=data.x)
    np.save(file=path + name + "_features.npy", arr=data.features)
    np.save(file=path + name + name + "_y.npy", arr=data.y)


def unformat_for_splitting(data: np.ndarray, initial_size):
    x, y, z = data.shape
    if initial_size < z:
        y_initial = y - 1
        X1, X2 = np.zeros(shape=(x, y_initial, z)), np.zeros(shape=(x, initial_size))
        for i in range(x):
            # X1[i], X2[i] = np.vsplit()
            X1[i], X2[i] = data[i][:y_initial], data[i][y_initial][:initial_size]
    else:
        raise Exception("Initial_size must be larger than z.")
    return X1, X2


def format_for_splitting(*args):
    X1, X2 = args[0], args[1]
    x, y, z = X1.shape
    n = pad_sequences(X2, maxlen=z, padding="post", value=0)
    result = np.zeros(shape=(x, y + 1, z))
    for i in range(x):
        result[i] = np.vstack((X1[i], n[i]))
    return result


def string2num(lists, unique_word):
    return [unique_word.get(l) for l in lists]


def flatting(sentences):
    return [word for sentence in sentences for word in sentence]


def margin(sentences, batch_size):
    """Permet d'ajouter du marge sur les bords.

    Args:
        sentences (list[list]): _description_
        batch_size (int): _description_

    Returns:
        list[list]: _description_

    Example:
        input: ['Peter', 'Blackburn']
        output ['<pad>', 'Blackburn', 'Peter', 'Blackburn', '<pad>']
    """
    batch_size = batch_size + 1
    b_size = int(batch_size // 2)
    pad = [np.zeros(shape=sentences[0][0].shape)]

    def __pad(sentence: list):
        n = len(sentence)
        if n <= b_size:
            sentence = sentence + pad * (b_size - n + 1)
            n = len(sentence)
        sentence = (
            list(reversed(sentence[1 : b_size + 1]))
            + sentence
            + list(reversed(sentence[n - b_size - 1 : n - 1]))
        )
        n = len(sentence)
        Sentences = []
        for i in range(b_size, n - b_size):
            Sentences.append(
                np.array(sentence[i - b_size : i + b_size + 1][1:], dtype="float32")
            )

        return Sentences

    Sentences = [__pad(sentence.copy()) for sentence in sentences]
    Sentences = [
        [Sentences[i][j] for j in range(len(sentences[i]))]
        for i in range(len(sentences))
    ]
    return Sentences


def zip_2D(*args):
    zipdata = list(zip(args[0], args[1], args[2], args[3], args[4], args[5]))
    zipdata = [
        list(zip(data[0], data[1], data[2], data[3], data[4], data[5]))
        for data in zipdata
    ]
    return zipdata


def unzip_2D(args):
    words, ners, chunks, poss, features, positions = [], [], [], [], [], []
    for arg in args:
        word, ner, chunk, pos, feature, position = [], [], [], [], [], []
        for triple in arg:
            word.append(triple[0])
            ner.append(triple[1])
            chunk.append(triple[2])
            pos.append(triple[3])
            feature.append(triple[4])
            position.append(triple[5])

        words.append(word)
        ners.append(ner)
        chunks.append(chunk)
        poss.append(pos)
        features.append(feature)
        positions.append(position)
    return words, ners, chunks, poss, features, positions


def padding(data: Data):
    data.flatten()
    data.x = data.sentences_num
    data.y = data.ner_tags
    data.gather()
    data.sentences_num = data.x
    data.ner_tags_num = data.y


def evaluation(y_true, y_predict, tag_noEntity:int):
    """ " noEnt == No Entities, ent == entities"""
    ent_true, ent_false, noEnt_true, noEnt_false = 0, 0, 0, 0
    x, y = y_true.shape
    for i in range(x):
        real_tag = np.argmax(y_true[i])
        predict_tag = np.argmax(y_predict[i])
        if predict_tag == tag_noEntity:
            if predict_tag == real_tag:
                noEnt_true += 1
            else:
                noEnt_false += 1
        else:
            if real_tag == predict_tag:
                ent_true += 1
            else:
                ent_false += 1
    print("----------------------- Evaluation -------------------------")
    if ent_true + ent_false == 0: 
        correct_pourcent_ent =  0
        incorrect_pourcent_ent = 0
    else: 
        correct_pourcent_ent = round(ent_true / (ent_true + ent_false), 3)
        incorrect_pourcent_ent = round(ent_false / (ent_false + ent_true), 3)
    if noEnt_false + noEnt_true == 0: 
        correct_pourcent_noEnt = 0
        incorrect_pourcent_noEnt = 0
    else: 
        correct_pourcent_noEnt = round(noEnt_true / (noEnt_true + noEnt_false), 3)
        incorrect_pourcent_noEnt = round(noEnt_false / (noEnt_false + noEnt_true), 3)
    print(
        "Entities: \n\t[correct=({}, {}%), incorrect=({}, {}%)]".format(
            ent_true, correct_pourcent_ent,
            ent_false, incorrect_pourcent_ent
        ),
        end="\n",
    )
    print(
        "No Entities: \n\t[correct=({}, {}%), incorrect=({}, {}%)]".format(
            noEnt_true, correct_pourcent_noEnt,
            noEnt_false, incorrect_pourcent_noEnt
        )
    )


def checkDataset(train: Data = None, test: Data = None, valid: Data = None):
    if train != None:
        print(
            "X_train",
            train.x.shape,
            "Features_train",
            train.features.shape,
            "y_train",
            train.y.shape,
            "\n",
        )
    if test != None:
        print(
            "X_test",
            test.x.shape,
            "Features_test",
            test.features.shape,
            "y_test",
            test.y.shape,
            "\n",
        )
    if valid != None:
        print(
            "X_valid",
            valid.x.shape,
            "Features_valid",
            valid.features.shape,
            "y_valid",
            valid.y.shape,
        )
