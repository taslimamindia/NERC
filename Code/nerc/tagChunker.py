from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk import pos_tag

def conll_tag_chunks(chunk_sents):
    tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff


# Define the chunker class
class NGramTagChunker(ChunkParserI):
    def __init__(self, train_sentences, tagger_classes=[UnigramTagger, BigramTagger]):
        train_sent_tags = conll_tag_chunks(train_sentences)
        self.chunk_tagger = combined_tagger(train_sent_tags, tagger_classes)

    def parse(self, tagged_sentence):
        if not tagged_sentence:
            return None
        pos_tags = [tag for word, tag in tagged_sentence]
        chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
        chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tags]
        wpc_tags = [
            (word, pos_tag, chunk_tag)
            for ((word, pos_tag), chunk_tag) in zip(tagged_sentence, chunk_tags)
        ]
        listes = tree2conlltags(conlltags2tree(wpc_tags))
        return list(zip(*listes))

    def parse_sents(self, sents):
        sentences, postags, chunks = [], [], []
        # [(list(sentence), list(postag), list(chunk)) for sentence, postag, chunk in self.parse(pos_tag(sent))]
        for sent in sents:
            sentence, postag, chunk = self.parse(pos_tag(sent))
            sentences.append(list(sentence))
            postags.append(list(postag))
            chunks.append(list(chunk))
        return sentences, postags, chunks
