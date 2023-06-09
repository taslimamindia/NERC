from data import Data
from loading import Loading

load_train = Loading("E:/PFE/CoNLL2003/NERC/Data/conll2003_english/train.txt")
load_test = Loading("E:/PFE/CoNLL2003/NERC/Data/conll2003_english/test.txt")
train, test = load_train.data, load_test.data
data = train + test

from word2vec import Model_Word2Vec

w2v = Model_Word2Vec(data.sentences)
# print(w2v.wv("EU"))

from base import Base_Model
m_base = Base_Model(data, w2v)
m_base.change(50, 300, 10)
m_base.preprocessing()
m_base.vectorization()
print(train.x.shape, train.y.shape)
