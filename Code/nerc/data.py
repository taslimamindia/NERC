from keras.utils import to_categorical


class Data:
    def __init__(self, VOCAB_SIZE = 100, PADDING_SIZE = 10, NUM_FILTERS = 256, KERNEL_SIZE = 3, DROPOUT_RATE = 0.2, BATCH_SIZE = 32, EPOCHS = 40):
        # # Dictionaries
        self.unique_ner_tags = {}
        self.unique_chunk_tags = {}
        self.unique_pos_tags = {}
        self.toCategorizeNerTags = {}
        self.toCategorizeChunkTags = {}
        self.toCategorizePosTags = {}
        # # Parameters
        self.VOCAB_SIZE = VOCAB_SIZE
        self.PADDING_SIZE = PADDING_SIZE
        self.NUM_FILTERS = NUM_FILTERS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.DROPOUT_RATE = DROPOUT_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        # # Listes
        self.sentences = []
        self.sentences_num = []
        self.ner_tags = []
        self.ner_tags_num = []
        self.chunk_tags = []
        self.pos_tags = []
        self.features = []
        self.positions = []
        # # Numpy arrays
        self.x, self.y = None, None

    def setParams(self, data):
        self.unique_ner_tags = data.unique_ner_tags
        self.unique_chunk_tags = data.unique_chunk_tags
        self.unique_pos_tags = data.unique_pos_tags
        self.toCategorizeNerTags = data.toCategorizeNerTags
        self.VOCAB_SIZE = data.VOCAB_SIZE
        self.PADDING_SIZE = data.PADDING_SIZE
        self.NUM_FILTERS = data.NUM_FILTERS
        self.KERNEL_SIZE = data.KERNEL_SIZE
        self.DROPOUT_RATE = data.DROPOUT_RATE
        self.BATCH_SIZE = data.BATCH_SIZE
        self.EPOCHS = data.EPOCHS

    def get_positions(self):
        self.positions = [
            [[i, j] for j in range(len(self.sentences[i]))]
            for i in range(len(self.sentences))
        ]

    def remove_attributes(self):
        self.sentences_num = []
        self.ner_tags_num = []
        self.chunk_tags = []
        self.pos_tags = []
        self.features = []
        self.x, self.y = None, None

    def __add__(self, o):
        data = Data()
        data.sentences = self.sentences + o.sentences
        data.sentences_num = self.sentences_num + o.sentences_num
        data.ner_tags = self.ner_tags + o.ner_tags
        data.ner_tags_num = self.ner_tags_num + o.ner_tags_num
        data.chunk_tags = self.chunk_tags + o.chunk_tags
        data.pos_tags = self.pos_tags + o.pos_tags
        data.features = self.features + o.features
        return data

    def tag2idx(self, tag):
        return self.unique_ner_tags.get(tag, None)

    def idx2tag(self, index):
        for tag, value in self.unique_ner_tags.items():
            if index == value:
                return tag
        return None

    def __unicity_tag(self, dico: dict, toCategorizeDico: dict, listes: list):
        unique_word = set()
        [unique_word.update(tags) for tags in listes]
        numerics = range(len(unique_word))
        dico.update(zip(unique_word, numerics))
        toCategorizeDico.update(
            zip(numerics, to_categorical(numerics, len(unique_word)))
        )

    def unicity(self):
        self.__unicity_tag(
            self.unique_ner_tags, self.toCategorizeNerTags, self.ner_tags_num
        )
        self.__unicity_tag(
            self.unique_chunk_tags, self.toCategorizeChunkTags, self.chunk_tags
        )
        self.__unicity_tag(
            self.unique_pos_tags, self.toCategorizePosTags, self.pos_tags
        )

    def features_level(self):
        def is_capitalize(word):
            return len(word) > 1 and word[0].isupper() and word[1:].islower()

        features = [
            [
                [
                    is_capitalize(word),
                    word.isupper(),
                    word.islower(),
                    word.istitle(),
                    word.isdigit(),
                ]
                for word in sentence
            ]
            for sentence in self.sentences
        ]
        self.features = [
            [[int(f) for f in feat] for feat in feature] for feature in features
        ]
