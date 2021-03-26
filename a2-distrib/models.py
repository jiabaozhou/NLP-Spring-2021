# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *

num_of_classes = 2


class FFNN(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """

    def __init__(self, word_embeddings, inp, hid, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors))
        self.word_embeddings.weight.requires_grad = False
        self.V = nn.Linear(inp, hid)
        self.V2 = nn.Linear(hid, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.dropout = nn.Dropout(0.3)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data; this is the sentence tensor.
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        # print(len(x[0]), x[0])
        embeddings = self.word_embeddings(x)
        # print(embeddings)
        mean = torch.mean(embeddings, dim=1).float()
        # torch.unsqueeze(mean, 1)
        # print(mean.shape)
        return self.W(self.g(self.V2(self.g(self.V(mean)))))


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, word_embeddings: WordEmbeddings):
        SentimentClassifier.__init__(self)
        self.word_embeddings = word_embeddings
        self.indexer = self.word_embeddings.word_indexer
        self.input = self.word_embeddings.get_embedding_length()
        self.hidden = 256
        self.output = num_of_classes
        self.model = FFNN(word_embeddings, self.input, self.hidden, self.output)

    def indexing(self, sentence: List[str]):
        sentence_idx = []
        for word in sentence:
            index = max(1, self.indexer.index_of(word))
            sentence_idx.append(index)
        return sentence_idx

    def predict(self, ex_words: List[str]):
        '''

        Args:
            ex_words: the sentence, a list of strings, or words in the sentence

        Returns:

        '''

        sentence_idx = self.indexing(ex_words)
        # sentence_tensor = torch.tensor(sentence_idx)
        probability = self.model.forward(torch.tensor([sentence_idx]))
        return torch.argmax(probability)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    max_len = -1
    for ex in train_exs:
        if len(ex.words) > max_len:
            max_len = len(ex.words)

    print(max_len)
    epochs = 15
    pad_size = max_len
    batch_size = 256
    clf = NeuralSentimentClassifier(word_embeddings)
    initial_learning_rate = 0.01
    optimizer = optim.Adam(clf.model.parameters(), lr=initial_learning_rate)
    batch_x = []
    batch_y = []
    for epoch in range(epochs):
        random.shuffle(train_exs)
        total_loss = 0.0
        for ex in train_exs:
            if len(batch_x) < batch_size:
                sentence_idx = clf.indexing(ex.words)
                padded_sentence_idx = [0] * pad_size
                padded_sentence_idx[:len(sentence_idx)] = sentence_idx
                label = ex.label
                batch_x.append(padded_sentence_idx)
                batch_y.append(label)
            else:
                target = torch.tensor(batch_y)
                clf.model.zero_grad()
                probability = clf.model.forward(torch.tensor(batch_x))
                loss = clf.model.loss_function(probability, target)
                total_loss += loss
                loss.backward()
                optimizer.step()
                batch_x = []
                batch_y = []
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return clf
