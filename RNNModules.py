#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from torch.utils.data import Dataset, DataLoader


"""
Created on Fri Dec 29 18:26:05 2017

@author: jkr
"""


use_cuda = torch.cuda.is_available()

"""
Created on Tue Dec 26 11:39:02 2017


@author: jkr
"""

def LocalizeAttn(x, width, input_dim, L):
    candidate = L*nn.Linear(input_dim, 1)(x)+L
    if use_cuda:
        candidate = candidate.cuda()
    ##Need to come up with a way to ensure this is in the right range...
    center = np.array(candidate.data)[0][0]
    weightvector = Variable(torch.Tensor([np.maximum(l-center+width, 0)*np.maximum(-l+center+width, 0)/width**2 for l in range(input_dim)]))
    if use_cuda:
        weightvector = weightvector.cuda()
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
        hidden = [hidden[0].view(1, 1, self.hidden_size), 
                  hidden[1].view(1, 1, self.hidden_size)]
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
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length):
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
        hidden = [hidden[0].view(1, 1, self.hidden_size), 
                  hidden[1].view(1, 1, self.hidden_size)]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0].view(1, self.hidden_size)), 1)), dim=1)
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
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length, L=2):
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

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(1, 1, self.hidden_size), 
                  hidden[1].view(1, 1, self.hidden_size)]
        
        attn_weights = self.attn(torch.cat((embedded[0], hidden[0].view(1, self.hidden_size)), 1))
        local_attn = LocalizeAttn(attn_weights, width=5, input_dim=self.max_length)
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