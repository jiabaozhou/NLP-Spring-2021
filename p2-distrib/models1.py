import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch import optim
from torch.autograd import Variable as Var, Variable
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
from torch.nn import functional as F


def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """

    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap / float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.0,
                 bidirect=True, teacher_forcing_prob=0.8):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.teacher_forcing_prob = teacher_forcing_prob
        self.hidden_size = hidden_size

        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        self.decoder = AttentionDecoder(emb_dim, hidden_size, len(output_indexer))
        self.loss_func = nn.NLLLoss(ignore_index=0)

    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, inp, batch_size):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len] vector of indices or a batched input/output
        [batch size x sent len]. y_tensor contains the gold sequence(s) used for training
        :param inp_lens_tensor/out_lens_tensor: either a vector of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        enc_output, context_mask, h_t = self.encode_input(x_tensor, inp_lens_tensor)
        # print(enc_output.shape, context_mask.shape, h_t[0].shape)
        hidden, cn = h_t
        # print(inp.shape)
        outputs = torch.zeros(out_lens_tensor, batch_size, len(self.output_indexer))
        # print(inp)
        for i in range(out_lens_tensor):
            output, hidden = self.decoder(inp, hidden, enc_output, context_mask, True) if i == 0 else self.decoder(inp,
                                                                                                                   hidden,
                                                                                                                   enc_output,
                                                                                                                   context_mask)
            # print(output.shape)
            is_teacher_forced = True if \
                random.choices([1, 0], [self.teacher_forcing_prob, 1 - self.teacher_forcing_prob])[
                    0] == 1 else False
            if is_teacher_forced:
                inp = y_tensor[:, i]
            else:
                inp = torch.argmax(output, dim=1)
                # print(inp)
            # if i % 32 == 0:
            #
            #     print(i, inp, is_teacher_forced)
            outputs[i] = output
            # print(outputs.shape)

        return outputs

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        self.eval()
        answer = []
        for i, ex in enumerate(test_data):
            x = np.array([ex.x_indexed])
            x_tensor = torch.from_numpy(x).long()
            inp_lens_tensor = torch.from_numpy(np.sum(x != 0, axis=1)).long()
            # print('this is ', i, inp_lens_tensor)
            enc_outputs, context_mask, h_t = self.encode_input(x_tensor, inp_lens_tensor)
            hn, cn = h_t
            token = "<SOS>"
            input = torch.full([1], self.output_indexer.index_of(token)).long()
            idx = 0
            predicted = []
            while token != "<EOS>" and idx < 70:
                # print(i, input.shape, input)
                output, hn = self.decoder(input, hn, enc_outputs, context_mask)
                # print(output)
                input = torch.argmax(output, dim=2).squeeze(0)
                # print(input)
                prob = torch.max(output)
                token = self.output_indexer.get_object(input.item())
                if token != "<EOS>":
                    predicted.append(token)
                idx += 1
            # print(predicted)
            answer.append([Derivation(ex, prob, predicted)])

        return answer

    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple). ONLY THE ENCODER FINAL
        STATES are needed for the basic seq2seq model. enc_output_each_word is needed for attention, and
        enc_context_mask is needed to batch attention.

        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        enc_output_each_word, enc_context_mask, enc_final_states = self.encoder(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        # print('lest go', enc_final_states_reshaped[1].shape)
        return enc_output_each_word, enc_context_mask, enc_final_states_reshaped


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """

    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNDecoder(nn.Module):
    def __init__(self, emb_dim, hidden, output, embedding_dropout=0):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(output, emb_dim)
        self.W = nn.Linear(hidden, output)
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=False)
        self.softmax = nn.Softmax()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_seq, hidden, cell):
        embeddings = self.embedding(input_seq.unsqueeze(0))
        output, (hn, cn) = self.rnn(embeddings, (hidden, cell))
        return self.W(output), hn, cn


class AttentionDecoder(nn.Module):
    def __init__(self, emb_dim, hidden, output, embedding_dropout=0):
        super(AttentionDecoder, self).__init__()
        self.embedding = EmbeddingLayer(emb_dim, output, embedding_dropout)
        # self.dropout = nn.Dropout(embedding_dropout)
        self.W = nn.Linear(hidden, output)
        self.rnn = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=False)
        self.attn = nn.Linear(hidden, hidden)
        self.combine = nn.Linear(hidden * 2, hidden)
        self.softmax = nn.Softmax(dim=1)

    def scoring(self, encoder_output, hidden, mask):
        """
        :param encoder_output: [input_seq_len x batch_size x hidden_size]
        :param hidden: [1 x batch_size x hidden_size]
        :param mask: [batch_size x input_seq_len]
        :return:

        scores: [batch_size x input_seq_len]

        """
        scores = torch.bmm(hidden.transpose(0, 1), self.attn(encoder_output).permute(1, 2, 0)) # [batch_size x 1 x input_seq_len]
        scores = scores.squeeze(1)
        # print(scores.shape)
        scores = scores.masked_fill_(mask==0, -1e18)
        return self.softmax(scores)

    def forward(self, input_seq, hidden, encoder_outputs, mask):
        embeddings = self.embedding(input_seq.unsqueeze(0))
        # print(embeddings.shape)
        # print(hidden.shape)
        # attentions = F.softmax(torch.cat((embeddings, hidden))) # self.scoring(encoder_outputs, output, mask)
        output, hidden = self.rnn(embeddings, hidden)
        # print(hidden.shape, encoder_outputs.shape, mask.shape)
        attentions = self.scoring(encoder_outputs, hidden, mask)
        # print('attentions', attentions.shape, encoder_outputs.shape)
        context = torch.bmm(attentions.unsqueeze(1), encoder_outputs.transpose(0, 1)).transpose(0, 1)
        # print('context', context.shape, context)
        # print('output', output.shape, output)
        probs = self.W(self.combine(torch.cat((context, hidden), 2)))
        # print('probs', probs.shape, probs)
        # input()
        return probs, hidden


# class AttentionDecoder(nn.Module):
#     def __init__(self, emb_dim, hidden, output, embedding_dropout=0):
#         super(AttentionDecoder, self).__init__()
#         self.embedding = nn.Embedding(output, emb_dim)
#         self.dropout = nn.Dropout(embedding_dropout)
#         self.W = nn.Linear(hidden, output)
#         self.rnn = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=False)
#         self.attn = nn.Linear(hidden, hidden)
#         self.combine = nn.Linear(hidden * 2, hidden)
#         self.softmax = nn.Softmax(dim=1)
#
#     def scoring(self, encoder_output, hidden, mask):
#         """
#         :param encoder_output: [input_seq_len x batch_size x hidden_size]
#         :param hidden: [1 x batch_size x hidden_size]
#         :param mask: [batch_size x input_seq_len]
#         :return:
#
#         scores: [batch_size x input_seq_len]
#
#         """
#         # print(mask.shape)
#         # encoder_out_len, batch_size, _ = encoder_output.size()
#         # scores = torch.zeros(batch_size, encoder_out_len)
#         # print(hidden.shape, encoder_output.shape)
#         scores = torch.bmm(hidden.transpose(0, 1), self.attn(encoder_output).permute(1, 2, 0)) # [batch_size x 1 x input_seq_len]
#         # print(scores.shape, scores)
#         # print(mask, scores.squeeze(1))
#         scores = scores.squeeze(1) * mask
#         # scores = scores.squeeze(1)
#         # print(scores)
#         # print("softmax'd scores", self.softmax(scores).shape, self.softmax(scores))
#         # input()
#         return self.softmax(scores)
#
#     def forward(self, input_seq, hidden, encoder_outputs, mask):
#         # print(input_seq.shape)
#         embeddings = self.embedding(input_seq.unsqueeze(0))
#         embeddings = self.dropout(embeddings)
#         # print(embeddings.shape)
#
#         attentions = F.softmax(torch.cat((embeddings, hidden))) # self.scoring(encoder_outputs, output, mask)
#         output, hidden = self.rnn(embeddings, hidden)
#         # attn_weights =
#         # print(output, hn)
#
#         # attentions = self.scoring(encoder_outputs, output, mask)
#         # print('attentions', attentions.shape, attentions)
#         context = torch.bmm(attentions.unsqueeze(1), encoder_outputs.transpose(0, 1)).transpose(0, 1)
#         # print('context', context.shape, context)
#         # print('output', output.shape, output)
#         probs = self.W(self.combine(torch.cat((context, hidden), 2)))
#         # print('probs', probs.shape, probs)
#         # input()
#         return probs, hidden


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """

    def __init__(self, input_emb_dim: int, hidden_size: int, bidirect: bool):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_emb_dim, hidden_size, num_layers=1, batch_first=True,
                           dropout=0., bidirectional=self.bidirect)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray(
            [[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        Note that output is only needed for attention, and context_mask is only used for batched attention.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True,
                                                             enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        # print('here', packed_embedding.size())
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = max(input_lens.data).item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        output = self.reduce_h_W(output)
        # print(output.shape, h_t[0].shape)
        return output, context_mask, h_t


###################################################################################################################
# End optional classes
###################################################################################################################


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int,
                             reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array(
        [[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)]
         for ex in exs])


def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer,
                       args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    hidden_size = 512
    batch_size = 20
    embedding_size = 512
    epochs = 20
    model = Seq2SeqSemanticParser(input_indexer, output_indexer, embedding_size, hidden_size)
    initial_lr = 0.002
    enc_optm = optim.Adam(model.encoder.parameters(), lr=initial_lr)
    dec_optm = optim.Adam(model.decoder.parameters(), lr=initial_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        # random.shuffle(all_train_input_data)
        z = list(zip(all_train_input_data, all_train_output_data))
        random.shuffle(z)
        all_train_input_data[:], all_train_output_data[:] = zip(*z)

        batch_index = batch_size
        while batch_index < len(all_train_input_data):
            # enc_optm.zero_grad()
            # dec_optm.zero_grad()
            batch_x = all_train_input_data[batch_index - batch_size:batch_index]
            batch_y = all_train_output_data[batch_index - batch_size:batch_index]
            x_tensor = torch.from_numpy(batch_x).long()
            y_tensor = torch.from_numpy(batch_y).long()
            inp_lens_tensor = torch.from_numpy(np.sum(batch_x != 0, axis=1)).long()
            out_lens_tensor = torch.from_numpy(np.sum(batch_y != 0, axis=1)).long()

            loss = batching(x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, model.encoder, model.decoder, enc_optm, dec_optm, output_max_len, batch_size, model.output_indexer, criterion, model)
            # inp = torch.full([batch_size], output_indexer.index_of("<SOS>")).long()
            # # print(inp)
            # outputs = model(x_tensor, inp_lens_tensor, y_tensor, output_max_len, inp, batch_size)
            # loss = criterion(outputs.view(batch_size*output_max_len, len(output_indexer)), y_tensor.view(batch_size*output_max_len))
            # loss.backward()
            # enc_optm.step()
            # dec_optm.step()

            total_loss += loss.item()
            batch_index += batch_size

        print(total_loss)
    # First create a model. Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters
    return model


def batching(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, enc_optm, dec_optm,
             output_max_len, batch_size, output_indexer, criterion, model, tfr=0.5):
    SOS_tok = output_indexer.index_of("<SOS>")
    EOS_tok = output_indexer.index_of("<EOS>")

    enc_optm.zero_grad()
    dec_optm.zero_grad()
    # criterion = nn.NLLLoss()
    # loss = 0
    enc_outputs, context_mask, enc_hidden = model.encode_input(input_batches, input_lengths)

    dec_inp = Variable(torch.LongTensor([SOS_tok] * batch_size))
    dec_hid = enc_hidden[0]

    max_target_length = max(target_lengths)

    dec_outputs = Variable(torch.zeros(output_max_len, batch_size, len(output_indexer)))

    teacher_forced = True if random.choices([1, 0], [tfr, 1 - tfr])[0] == 1 else False

    if teacher_forced:
        for i in range(max_target_length):
            dec_output, dec_hid = decoder(dec_inp, dec_hid, enc_outputs, context_mask)
            dec_outputs[i] = dec_output
            dec_inp = target_batches[:, i]
            # print(dec_inp, "teacher forced!!!!!!!!!!!!!")
    else:
        num_EOS = 0
        for i in range(max_target_length):
            dec_output, dec_hid = decoder(dec_inp, dec_hid, enc_outputs, context_mask)
            topv, topi = dec_output.data.topk(1)
            if EOS_tok in topi:
                num_EOS += 1
                if num_EOS == batch_size:
                    break
            # print(dec_output.shape, dec_output)
            dec_outputs[i] = dec_output
            dec_inp = torch.flatten(Variable(torch.LongTensor(topi.squeeze(1))))
            # print(dec_inp, "not teacher forced")

    # print(dec_outputs.shape, target_batches.shape)
    loss = criterion(dec_outputs.transpose(1, 2), target_batches.transpose(0, 1))
    loss.backward()

    enc_optm.step()
    dec_optm.step()
    return loss
