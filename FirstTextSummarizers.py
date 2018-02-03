#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from numpy import random
import torch
import torch.nn as nn
from torch.autograd import Variable
#import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F
#from torch.utils.data import Dataset
import PreprocessingNLPData
import pandas as pd
import pdb
import random

"""
Created on Mon Jan 15 10:50:25 2018

First PyTorch text summarizers, using Wikipedia 2010 corpus

@author: jkr
"""

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

class DataStatistics:
    def __init__(self, name, max_target_vocab=20000):
        self.name = name
        self.targetword2index = {"SOS":0, "EOS":1,'<unk>':2}
        self.targetword2count = {'<unk>':100, 'SOS':100, 'EOS':100}
        self.targetindex2word = {0: "SOS", 1: "EOS", 2:'<unk>'}
        self.n_words_target = 3  # Count SOS and EOS
        self.max_length = 0
        self.glove_dict = create_glove_dict()
        self.glove_vector_size = len(self.glove_dict['the'])
        self.max_target_vocab = max_target_vocab

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)
                
    def updateMaxLength(self, sentence):
        for word in sentence:
            if len(sentence)+3>self.max_length:
                self.max_length = len(sentence)+3
                print(str(len(sentence)+3))

    def addWord(self, word):
        word = word.lower()
        if word not in self.targetword2index:
            self.targetword2index[word] = self.n_words_target-2
            self.targetword2count[word] = 1
            self.targetindex2word[self.n_words_target] = word.lower()
            self.n_words_target += 1
        else:
            self.targetword2count[word] += 1
        
        
    ##This is so fucking poorly implemented I can barely believe I 
    ##wrote it. You obviously need to rewrite SO YOU DON'T LOOP OVER ALL 
    ##THE VOCAB A BILLION TIMES!!!
    
    def restrictVocab(self):
        i2w = pd.DataFrame.from_dict(self.targetindex2word,orient='index').reset_index(drop=False)
        i2w.columns=['Idx', 'Word']
        w2c = pd.DataFrame.from_dict(self.targetword2count,orient='index').reset_index(drop=False)
        w2c.columns=['Word', 'Count']
        df = i2w.merge(w2c, how = 'inner', on='Word')
        df.sort_values(by='Count', ascending=False, inplace=True)
        restricted = df.head(self.max_target_vocab)
        restricted.reset_index(inplace=True, drop=False)
        i2w = {idx:row['Word'] for idx, row in restricted.iterrows()}
        w2c = {row['Word']:row['Count'] for idx, row in restricted.iterrows()}
        w2i = {row['Word']:idx for idx, row in restricted.iterrows()}
        self.targetindex2word = i2w
        self.targetword2index = w2i
        self.targetword2count = w2c
        self.n_words_target = self.max_target_vocab
        
def create_glove_dict():
    glove_dict={}
    with open('/home/jkr/GloVe-1.2/vectors.txt') as file:
        for line in file:
            ls = line.split()
            word=ls[0]
            ls.pop(0)
            vec = ls.copy()
            idx = 0
            for w in vec:
                vec[idx] = float(w)
                idx+=1
            glove_dict[word] = vec
    return glove_dict

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bi=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.linearinputtrans = nn.Linear(input_size, hidden_size)
        self.bi=bi
        if bi:
            self.lstm = nn.LSTM(hidden_size, int(hidden_size/2),
                                bidirectional=bi, batch_first=True)
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bi,
                                batch_first=True)

    def forward(self, input, hidden):
        output = self.linearinputtrans(input)
        for i in range(self.n_layers):
            x = output.clone()
            output, hidden = self.lstm(output,
                     hidden)
            output = output+x
        return output, hidden

    def initHidden(self, batch_size):
        if self.bi:
            result = Variable(torch.zeros(2, batch_size, int(self.hidden_size/2)))
        else:
            result = Variable(torch.zeros(1, batch_size, self.hidden_size))
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
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_ouputs):
        output = self.embedding(input).view(1, 1, -1)
        hidden = [hidden[0].view(1, 1, self.hidden_size), 
                  hidden[1].view(1, 1, self.hidden_size)]
        for i in range(self.n_layers):
            output = F.relu(output)
            x = output.clone()
            output, hidden = self.lstm(output,
                     hidden)
            if i>0 and i<self.n_layers-1:
                output=self.dropout(output)
            output = output+x
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, *, hidden_size, output_size, max_length, 
                 n_layers=1, dropout_p=0.1):
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
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1, self.hidden_size)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(-1, 1, self.hidden_size), 
                  hidden[1].view(-1, 1, self.hidden_size)]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs).squeeze(1)

        output = torch.cat((embedded, attn_applied),1)
        output = self.attn_combine(output).unsqueeze(0)
        
        hidden = [hidden[0].view(1, -1, self.hidden_size), 
                  hidden[1].view(1, -1, self.hidden_size)]

        for i in range(self.n_layers):
            output = F.relu(output).unsqueeze(1).view(-1, 1, self.hidden_size)
            x = output.clone()
            output, hidden = self.lstm(output,
                     hidden)
            output = output+x
        
#        pdb.set_trace()
        
        output = F.log_softmax(self.out(output), dim=-1)
        return output, hidden
    
    def initHidden(self,batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
        
class LocalAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, 
                 n_layers=1, dropout_p=0.1, L=2):
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
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn_linear = nn.Linear(self.max_length, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(1, 1, self.hidden_size), 
                  hidden[1].view(1, 1, self.hidden_size)]
        
        attn_weights = self.attn(torch.cat((embedded[0], hidden[0].view(1, self.hidden_size)), 1))
        local_attn = self.LocalizeAttn(attn_weights, width=5, input_dim=self.max_length)
        attn_applied = torch.bmm(local_attn.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            
            output = F.relu(output)
            x = output.clone()
            output, hidden = self.lstm(output, hidden)
            output = output+x

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
    def LocalizeAttn(self, x, width, input_dim):
        candidate = self.attn_linear(x)
        ##Need to come up with a way to ensure this is in the right range...
        center = np.array(candidate.data)[0][0]
        weightvector = Variable(torch.Tensor([np.maximum(l-center+width, 0)*np.maximum(-l+center+width, 0)/width**2 for l in range(input_dim)])).cuda()
        return x*weightvector


def batchedSeq2SeqTrain(data_statistics,input_variables, target_variables, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion):
    """General function for online training
    of sequence-to-sequence models
    """
    
    input_variables.sort(key = len)
    input_variables.reverse()
    target_variables.sort(key = len)
    target_variables.reverse()
    
    batch_size = int(len(input_variables))
    assert batch_size == int(len(target_variables))
    
    input_lengths = [input_variable.size()[0] for input_variable in input_variables]
    target_lengths = [target_variable.size()[0] for target_variable in target_variables]

    max_input_length = int(np.max(input_lengths))
    max_target_length = int(np.max(target_lengths))

    encoder_outputs = Variable(torch.zeros(batch_size, max_input_length,
                                           encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    ##Padding input variables
    var_list = []
    for variable in input_variables:
        if variable.size()[0]<ds.max_length:
            diff = ds.max_length-variable.size()[0]
            var_size = list(variable.size())
            var_size[0] = int(diff)
            to_pad = Variable(torch.zeros(*var_size))
            to_pad = to_pad.cuda() if use_cuda else to_pad
            var_list.append(torch.cat([variable, to_pad]))
        else:
            var_list.append(variable)
    input_variables = torch.cat(var_list).view(batch_size, ds.max_length, -1)
    ##Packing these padded variables
    #batched_input = torch.nn.utils.rnn.pack_padded_sequence(input_variables, input_lengths, batch_first=True)
    
    output_var_list = []
    for variable in target_variables:
        if variable.size()[0]<max_target_length:
            diff = max_target_length-variable.size()[0]
            var_size = list(variable.size())
            var_size[0] = int(diff)
            to_pad = Variable(torch.LongTensor(np.zeros(var_size)))
            to_pad = to_pad.cuda() if use_cuda else to_pad
            output_var_list.append(torch.cat([variable, to_pad]))
        else:
            output_var_list.append(variable)
    target_variables = torch.cat(output_var_list).view(batch_size, int(max_target_length), -1)
    ##Packing these padded variables
    #batched_target = torch.nn.utils.rnn.pack_padded_sequence(target_variables, target_lengths, batch_first=True)
    
    
    encoder_hx = encoder.initHidden(batch_size)
    encoder_cx = encoder.initHidden(batch_size)
    encoder_hidden = [encoder_hx, encoder_cx]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    encoder_outputs = Variable(torch.zeros(batch_size, ds.max_length,
                                           encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    
    for ei in range(ds.max_length):
        encoder_output, encoder_hidden = encoder(
            input_variables[:,ei,:].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei, :] = encoder_output[:,0,:]

    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size])).view(batch_size, -1)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden.clone()
    
    teacher_forcing_ratio = .5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            target_vars = target_variables.squeeze(2)[:,di+1]
            loss += criterion(decoder_output.squeeze(1), target_vars)
            
            
            decoder_input = target_vars  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            preds = decoder_output.data.topk(1)
            ni = torch.cat(preds[1])

            decoder_input = Variable(ni)
            target_vars = target_variables.squeeze(2)[:,di+1]
            
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if len(target_vars[target_vars[:]>0])>0:
                loss += criterion(decoder_output.squeeze(1)[(target_vars[:]>0).nonzero().squeeze()],
                              target_vars[target_vars[:]>0])
            
    loss.backward()
    
    
    
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 1)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 1)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] 


def evaluate(ds, encoder, decoder, input_variable, max_decoder_length=100, batch_size=1):
    input_length = input_variable.size()[0]

    
    encoder_hx = encoder.initHidden(batch_size)
    encoder_cx = encoder.initHidden(batch_size)
    encoder_hidden = [encoder_hx, encoder_cx]
    
    encoder_outputs = Variable(torch.zeros(batch_size, ds.max_length,
                                           encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei].unsqueeze(0).unsqueeze(0), encoder_hidden)
        encoder_outputs[0,ei,:] = encoder_output[0]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    for di in range(max_decoder_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input.squeeze(0), decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0][0]
        
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            try:
                word = ds.targetindex2word[ni]
            except:
                word = '<unk>'
            decoded_words.append(word)

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words
            
def word2index(ds, type, word):
    if type =='input':
        try:
            return ds.glove_dict[word.lower()]
        except:
            return ds.glove_dict['<unk>']
    else:
        try:
            return ds.targetword2index[word.lower()]
        except:
            return ds.targetword2index['<unk>']
    
def indexesFromSentence(ds, type, sentence):
    return [word2index(ds, type, word) for word in sentence]


def variableFromSentence(ds, type, sentence):
    
    indexes = indexesFromSentence(ds, type, sentence)
    #print(sentence)
    if type =='output':
        indexes.append(EOS_token)
        indexes.insert(0, SOS_token)
        result = Variable(torch.LongTensor(np.array(indexes)).view(-1, 1))
    elif type =='input':
        result = Variable(torch.FloatTensor(np.array(indexes)).view(-1, ds.glove_vector_size))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(ds, pair):
    if pair[0] and pair[1]:
        input_variable = variableFromSentence(ds, 'input', pair[1])
        target_variable = variableFromSentence(ds, 'output', pair[0])
        title = variableFromSentence(ds, 'output', [pair[0][0]])
        return (input_variable, target_variable, title)

def batchedTrainIters(data_statistics, pairs, encoder, decoder, n_iters, n_examples, batch_size=128, print_every=1000,
                      plot_every=100, learning_rate=1e-3):
    """Function to train general seq2seq models with batching
    """
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(data_statistics, (pair[0], pair[1])) for pair in pairs]
    random.shuffle(training_pairs)
    val_pairs = training_pairs[int(.99*len(training_pairs)):]
    training_pairs = training_pairs[:int(.99*len(training_pairs))]

    criterion = nn.NLLLoss()

    for iter in range(0, n_iters, batch_size):
        if iter%n_examples<(iter+batch_size)%n_examples:
            training_batch = training_pairs[iter%n_examples:(iter+batch_size)%n_examples]
            
        else:
            list1 = training_pairs[iter%n_examples:]
            list2 = training_pairs[:(iter+batch_size)%n_examples]
            training_batch = list1+list2
            
        if training_batch:
            input_variables = [example[0] for example in training_batch]
            target_variables = [example[1] for example in training_batch]
    
            loss = batchedSeq2SeqTrain(data_statistics, input_variables, target_variables, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time.time()-start,
                                         iter, iter / n_iters * (100*batch_size), print_loss_avg))
            print(data_statistics.targetindex2word[int(val_pairs[0][1][1])])
            print(evaluate(data_statistics, encoder, decoder, val_pairs[0][0]))
                
if __name__ == '__main__':
    Data = PreprocessingNLPData.WikipediaCorpusFirstMillion()
    training_pairs = Data.raw_data
    ds = DataStatistics('WikipediaCorpus', max_target_vocab=20000)
    for pair in training_pairs:
        ##0th element is the summary--1st is the long description.
        ##A little backwards in the opinion of some
        ds.addSentence(pair[0])
        ds.updateMaxLength(pair[1])
    ds.restrictVocab()
    hidden_size = 256
    
#    encoder0 = EncoderRNN(ds.n_words_target, hidden_size, n_layers=1)
#    attn_decoder0 = DecoderRNN(hidden_size, ds.n_words_target,
#                                   1, dropout_p=0)
#
#    
#    if use_cuda:
#        encoder0 = encoder0.cuda()
#        attn_decoder0 = attn_decoder0.cuda()
#    print("No attention")
#    trainIters(ds, training_pairs, encoder0, attn_decoder0, 10000, print_every=1)
###    
    
    encoder1 = EncoderRNN(ds.glove_vector_size, hidden_size, n_layers=4)
    attn_decoder1 = AttnDecoderRNN(hidden_size=hidden_size,
                                   output_size=ds.n_words_target,
                                   max_length=ds.max_length,
                                   n_layers=4, dropout_p=0)

    
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()
    print("Global attention")
#    trainIters(ds, training_pairs, encoder1, attn_decoder1, 1000, len(training_pairs), print_every=10)
    batchedTrainIters(data_statistics=ds,
                      pairs=training_pairs,
                      encoder=encoder1,
                      decoder=attn_decoder1,
                      n_iters=int(1e3),
                      n_examples=len(training_pairs),
                      batch_size=10,
                      print_every=100,
                      learning_rate = 1e-3)

#    encoder2 = EncoderRNN(ds.n_words_target, hidden_size, n_layers=1)
#    attn_decoder2 = LocalAttnDecoderRNN(hidden_size, ds.n_words_target, 
#                                        max_length=ds.max_length,
#                                        n_layers=1,
#                                        dropout_p=0)
#
#    
#    if use_cuda:
#        encoder2 = encoder2.cuda()
#        attn_decoder2 = attn_decoder2.cuda()
#    print("Local attention")
#    trainIters(ds, training_pairs, encoder2, attn_decoder2, 1000, print_every=1)