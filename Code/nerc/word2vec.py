from gensim.models import Word2Vec

class Model_Word2Vec:
    def __init__(self, sentences, vocab_size):
        self.model = Word2Vec(
            sentences=sentences, min_count=1, vector_size=vocab_size, window=5
        )

    def wv(self, word):
        return self.model.wv[word]