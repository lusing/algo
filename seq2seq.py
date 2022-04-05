import torch as t
import tensorflow as tf
import unicodedata
import re

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'BoS', 1:'EoS'}
        self.n_words = 2 # Count of BoS and EoS

    # 断句用
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print('Reading lines...')

    # Read the file and split into lines
    lines = open('data/\%s-\%s.txt' \% (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    #Split every line inot pairs and normalize
    pairs [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reverse(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = ("i am ", "he is")


class EncoderRNN(t.nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = t.nn.Embedding(input_size, hidden_size)
        self.gru = t.nn.GRU(hidden_size, hidden_size)


    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.embedding(input) # batch, hidden
        output = embedded.permute(1,0,2)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden


    def initHidden(self):
        result = t.autograd.Variable(t.zeros(1,1, self.hidden_size))
        return result


class DecoderRNN(t.nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = t.nn.Embedding(output_size, hidden_size)
        self.gru = t.nn.GRU(hidden_size, hidden_size)
        self.out = t.nn.Linear(hidden_size, output_size)
        self.softmax = t.nn.LogSoftmax()


    def forward(self, input, hidden):
        output = self.embedding(input) # batch, 1, hidden
        output = output.permute(1,0,2) # 1, batch, hidden
        for i in range(self.n_layers):
            output = t.nn.functional.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.softmax(self.out(output[0]))
        return output, hidden


    def initHidden(self):
        result = t.autograd.Variable(t.zeros(1,1,self.hidden_size))
        return result

