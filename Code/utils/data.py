class Data(object):
    unique_words = {"<PAD>":0}
    unique_ner_tags = {"O":0}
    MAX_LENGTH = 50
    VOCAB_SIZE = 50
    
    def __init__(self):
        self.sentences = []
        self.sentences_num = None
        self.ner_tags = []
        self.ner_tags_num = None
        self.chunk_tags = []
        self.pos_tags = []
        self.x, self.y = None, None  
    def word2idx(self, word:str):
        return Data.unique_words.get(word, None)
    def idx2word(self, index:int):
        for word, value in Data.unique_words.items():
            if index is value: return word
        return None    
    def tag2idx(self, tag):
        return Data.unique_ner_tags.get(tag, None)
    def idx2tag(self, index):
        for tag, value in Data.unique_ner_tags.items():
            if index == value: return tag
        return None
    def unicity(self):
        unique_sent, unique_tag = set(), set()
        [unique_tag.update(tags) for tags in self.ner_tags_num]
        [unique_sent.update(tags) for tags in self.sentences_num]
        max_tags = len(Data.unique_ner_tags)
        max_words = len(Data.unique_words)
        for word in list(unique_sent):
            if Data.unique_words.get(word, None) == None:
                Data.unique_words[word] = max_words
                max_words += 1
        for tag in list(unique_tag):
            if Data.unique_ner_tags.get(tag, None) == None:
                Data.unique_ner_tags[tag] = max_tags
                max_tags += 1

