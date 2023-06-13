class Parameter:
    def __init__(self, data):
        self.unique_ner_tags = data.unique_ner_tags.copy()
        self.unique_chunk_tags = data.unique_chunk_tags.copy()
        self.unique_pos_tags = data.unique_pos_tags.copy()
        self.toCategorizeNerTags = data.toCategorizeNerTags.copy()
        self.VOCAB_SIZE = data.VOCAB_SIZE
        self.PADDING_SIZE = data.PADDING_SIZE
        self.NUM_FILTERS = data.NUM_FILTERS
        self.KERNEL_SIZE = data.KERNEL_SIZE
        self.DROPOUT_RATE = data.DROPOUT_RATE
        self.BATCH_SIZE = data.BATCH_SIZE
        self.EPOCHS = data.EPOCHS