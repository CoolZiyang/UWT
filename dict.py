class Dictionary(object):

    def __init__(self, id2word, word2id, lang):
        self.id2word = id2word
        self.word2id = word2id
        self.lang = lang
        self.check_valid()

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, i):
        return self.id2word[i]

    def __contains__(self, w):
        return w in self.word2id

    def __eq__(self, y):
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return self.lang == y.lang and all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        assert len(self.id2word) == len(self.word2id), "index and reversed-index don't match"
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word):
        return self.word2id[word]

    def prune(self, max_voc):
        assert max_voc >= 1
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_voc}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.check_valid()

