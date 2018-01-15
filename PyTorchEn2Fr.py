#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
#import string
import re
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
#import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

use_cuda = torch.cuda.is_available()
##If this breaks it's a.s. a PATH/LD_LIBRARY_PATH thing...see
"""https://devtalk.nvidia.com/default/topic/536238/could-not-locate-devicequery-on-my-installation/
"""
print(use_cuda)

"""
Created on Tue Dec 26 11:39:02 2017


@author: jkr
"""

def LocalizeAttn(x, width, input_dim, MAX_LENGTH=MAX_LENGTH):
    candidate = MAX_LENGTH*nn.Linear(input_dim, 1)(x)+MAX_LENGTH
    ##Need to come up with a way to ensure this is in the right range...
    center = np.array(candidate.data)[0][0]
    weightvector = Variable(torch.Tensor([np.maximum(l-center+width, 0)*np.maximum(-l+center+width, 0)/width**2 for l in range(input_dim)])).cuda()
    return x*weightvector




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bi=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bi=bi

        self.embedding = nn.Embedding(input_size, hidden_size)
        if bi:
            self.lstm = nn.LSTM(hidden_size, int(hidden_size/2), bidirectional=bi)
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bi)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        if self.bi:
            result = Variable(torch.zeros(2, 1, int(self.hidden_size/2)))
        else:
            result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p=dropout_p


        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_ouputs):
        output = self.embedding(input).view(1, 1, -1)
        hidden = [hidden[0].view(1, 1, hidden_size), 
                  hidden[1].view(1, 1, hidden_size)]
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.lstm(output, hidden)
            if i>0 and i<self.n_layers-1:
                output=self.dropout(output)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(1, 1, hidden_size), 
                  hidden[1].view(1, 1, hidden_size)]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0].view(1, hidden_size)), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
        
class LocalAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH, L=2):
        super(LocalAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_localize = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn_linear = nn.Linear(self.max_length, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(1, 1, hidden_size), 
                  hidden[1].view(1, 1, hidden_size)]
        
        attn_weights = self.attn(torch.cat((embedded[0], hidden[0].view(1, hidden_size)), 1))
        local_attn = self.LocalizeAttn(attn_weights, width=5, input_dim=self.max_length)
        attn_applied = torch.bmm(local_attn.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
    def LocalizeAttn(self, x, width, input_dim, MAX_LENGTH=MAX_LENGTH):
        candidate = MAX_LENGTH*self.attn_linear(x)+MAX_LENGTH
        ##Need to come up with a way to ensure this is in the right range...
        center = np.array(candidate.data)[0][0]
        weightvector = Variable(torch.Tensor([np.maximum(l-center+width, 0)*np.maximum(-l+center+width, 0)/width**2 for l in range(input_dim)])).cuda()
        return x*weightvector
    
        
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


def seq2seqtrain(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """General function for online training
    of sequence-to-sequence models
    """
    encoder_hx = encoder.initHidden()
    encoder_cx = encoder.initHidden()
    encoder_hidden = [encoder_hx, encoder_cx]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    ##So you see here the GRU calls in PyTorch only implement the ONE UNIT,
    ##and you have explicitly the inability to parallelize.
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    
    teacher_forcing_ratio = 0.5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()
    
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 5)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def batchevaluate(val_pairs, encoder, decoder, criterion, max_length=MAX_LENGTH):
    """Notice this is online evaluation, which is super fucking slow--
    I just don't know how to vectorize this yet in PyTorch...
    """
    loss = 0
    for val_pair in val_pairs:
        input_variable = val_pair[0]
        target_variable = val_pair[1]
        encoder_hx = encoder.initHidden()
        encoder_cx = encoder.initHidden()
        encoder_hidden = [encoder_hx, encoder_cx]
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
    
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
        ##So you see here the RNN calls in PyTorch only implement the ONE UNIT,
        ##and you have explicitly the inability to parallelize.
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
    
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
        decoder_hidden = encoder_hidden
    
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
    
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=1e-3):
    """Function to train general seq2seq models in online fashion
    """
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    val_pairs = [variablesFromPair(input_lang, output_lang, pair)
    for pair in pairs[-100:]]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = seq2seqtrain(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print(batchevaluate(val_pairs, encoder,
                     decoder, criterion))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def swaggyevaluate(input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hx = encoder.initHidden()
    encoder_cx = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    #Again here you see explicitly the inability to parallelize
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hx, encoder_cx)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hx, decoder_cx = encoder_hx, encoder_cx

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hx, decoder_cx = decoder(
            decoder_input, decoder_hx, decoder_cx, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words

def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = swaggyevaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
        
class TranslationDataset(Dataset):
    def __init__(self, input_lang, output_lang):
        self.input_lang, self.output_lang, self.pairs = TranslationDataset.prepareData(input_lang, output_lang, True)
        
    def __len__(self):
        """Pairs is just a list so this can be implemented very simply
        """
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Pairs is a list
        """
        return self.pairs[idx]
    
    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Lowercase, trim, and remove non-letter characters
    
    
    def normalizeString(s):
        s = TranslationDataset.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    
    def readLangs(lang1, lang2, reverse=False):
        print("Reading lines...")
    
        # Read the file and split into lines
        lines = open('/home/jkr/Documents/MLData/Translation/%s2%s.txt' %(lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')
    
        # Split every line into pairs and normalize
        pairs = [[TranslationDataset.normalizeString(s) for s in l.split('\t')] for l in lines]
    
        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
    
        return input_lang, output_lang, pairs
    
    def filterPair(p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)
    
    
    def filterPairs(pairs):
        return [pair for pair in pairs if TranslationDataset.filterPair(pair)]
    
    def prepareData(lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = TranslationDataset.readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = TranslationDataset.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs

    
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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
            
        
if __name__ == '__main__':
    Data = TranslationDataset('eng', 'fra')
    input_lang, output_lang, pairs = Data.input_lang, Data.output_lang, Data.pairs
    print(random.choice(pairs))
    hidden_size = 256
    
#    encoder0 = EncoderRNN(input_lang.n_words, hidden_size, n_layers=2)
#    attn_decoder0 = DecoderRNN(hidden_size, output_lang.n_words,
#                                   2, dropout_p=0.1)
##    attn_decoder0.register_backward_hook(lambda grad: torch.clamp(grad, -5, 5))
#    
#    if use_cuda:
#        encoder0 = encoder0.cuda()
#        attn_decoder0 = attn_decoder0.cuda()
#    print("No attention")
#    trainIters(input_lang, output_lang, pairs, encoder0, attn_decoder0, 75000, print_every=5000)
##    
#    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, n_layers=2)
#    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
#                                   2, dropout_p=0.1)
##    attn_decoder1.register_backward_hook(lambda grad: torch.clamp(grad, -5, 5))
#    
#    if use_cuda:
#        encoder1 = encoder1.cuda()
#        attn_decoder1 = attn_decoder1.cuda()
#    print("Global attention")
#    trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 75000, print_every=5000)
#    
    encoder2 = EncoderRNN(input_lang.n_words, hidden_size, n_layers=2)
    attn_decoder2 = LocalAttnDecoderRNN(hidden_size, output_lang.n_words,
                                   2, dropout_p=0.1)
#    attn_decoder2.register_backward_hook(lambda grad: torch.clamp(grad, -5, 5))
    
    if use_cuda:
        encoder2 = encoder2.cuda()
        attn_decoder2 = attn_decoder2.cuda()
    print("Local attention")
    trainIters(input_lang, output_lang, pairs, encoder2, attn_decoder2, 75000, print_every=5000)