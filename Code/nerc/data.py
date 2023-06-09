class Data:
    unique_words = {"<PAD>": 0}
    unique_ner_tags = {"O": 0}
    unique_chunk_tags = {}
    unique_pos_tags = {}
    MAX_LENGTH = 50
    VOCAB_SIZE = 100
    PADDING_SIZE = 10
    # Hyperparameters
    NUM_FILTERS = 256
    KERNEL_SIZE = 3
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    EPOCHS = 40

    def __init__(self):
        self.sentences = []
        self.sentences_num = []
        self.ner_tags = []
        self.ner_tags_num = []
        self.chunk_tags = []
        self.pos_tags = []
        self.features = []
        self.positions = []
        self.x, self.y = None, None

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

    def word2idx(self, word: str):
        return Data.unique_words.get(word, None)

    def idx2word(self, index: int):
        for word, value in Data.unique_words.items():
            if index is value:
                return word
        return None

    def tag2idx(self, tag):
        return Data.unique_ner_tags.get(tag, None)

    def idx2tag(self, index):
        for tag, value in Data.unique_ner_tags.items():
            if index == value:
                return tag
        return None

    def __unicity_tag(self, dico: dict, listes: list):
        unique_word = set()
        [unique_word.update(tags) for tags in listes]
        max_index = len(dico)
        for word in list(unique_word):
            if dico.get(word, None) == None:
                dico[word] = max_index
                max_index += 1

    def unicity(self):
        self.__unicity_tag(Data.unique_ner_tags, self.ner_tags_num)
        self.__unicity_tag(Data.unique_words, self.sentences_num)
        self.__unicity_tag(Data.unique_chunk_tags, self.chunk_tags)
        self.__unicity_tag(Data.unique_pos_tags, self.pos_tags)

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
