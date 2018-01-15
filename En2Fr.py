#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
import numpy as np

"""
Created on Sun Dec 17 18:43:10 2017
Building LSTM RNN with multiple attention modules
@author: jkr
"""
def LocalRelu(x, width):
    return K.relu(x+width)*K.relu(-width-x)


def LocalAttention(base_layer, width, context):
    L = base_layer.output_shape
    return LocalRelu(L*Dense(L, activation='sigmoid')(context))

def VanillaLSTM3(num_encoder_tokens, num_decoder_tokens):
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_1 = LSTM(latent_dim, return_sequences=True)(encoder_inputs)
    encoder_2 = LSTM(latent_dim, return_sequences=True)(encoder_1)
    encoder_3, state_h, state_c= LSTM(latent_dim, return_sequences=False, 
                                      return_state=True)(encoder_2)
    encoder_state = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm1 = LSTM(latent_dim, return_sequences=True)
    decoder_1 = decoder_lstm1(decoder_inputs, 
                    initial_state=[state_h, state_c])
    decoder_lstm2 = LSTM(latent_dim, return_sequences=True)
    decoder_2 = decoder_lstm2(decoder_1)
    decoder_lstm3= LSTM(latent_dim, return_sequences=True, 
                           return_state=True)
    decoder_3, _, _ = decoder_lstm3(decoder_2)
    
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    
    decoder_outputs = decoder_dense(decoder_3)
    #I think some of the shit below may not be necessary?
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    inference_decoder_outputs, decoder_state_h, decoder_state_c =\
    decoder_lstm3(decoder_lstm2(decoder_lstm1(decoder_inputs)))
    decoder_states = [decoder_state_h, decoder_state_c]
    inference_decoder_predictions = decoder_dense(inference_decoder_outputs)
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs),\
Model(encoder_inputs, encoder_state), Model([decoder_inputs] + decoder_states_inputs,
      [inference_decoder_predictions] + decoder_states)

    
def BidirectionalLSTM3(num_encoder_tokens, num_decoder_tokens):
    
    pass
    

def PreprocessData(data_path, num_samples):
    input_texts = []
    input_texts.append('I love you')
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path).read().split('\n')
    for line in lines[:min(num_samples, len(lines)-1)]:
        input_text, target_text = line.split('\t')
        target_text = '\t'+target_text+'\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    input_texts.append('I love you')
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
    input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])
    
    encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype = 'float32')
    decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
    decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')    
    
    for i, (input_text, target_text) in enumerate(zip(input_texts,
           target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1
            if t>0:
                decoder_target_data[i, t-1, target_token_index[char]] = 1
    return input_texts, num_encoder_tokens, num_decoder_tokens,\
            encoder_input_data, decoder_input_data, decoder_target_data,\
            input_token_index, target_token_index, max_decoder_seq_length

batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
data_path = '/home/jkr/Documents/MLData/FraEngTranslation/fra.txt'

input_texts, num_encoder_tokens, num_decoder_tokens,\
encoder_input_data, decoder_input_data, decoder_target_data,\
input_token_index, target_token_index, max_decoder_seq_length = \
PreprocessData(data_path, num_samples)


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder0 = LSTM(latent_dim, return_sequences=True, return_state=False)
int_encoder = encoder0(encoder_inputs)
encoder1 = LSTM(latent_dim, return_sequences=True, return_state=False)
encoded1 = encoder1(int_encoder)
encoder2 = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder2(encoded1)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, callbacks=[es])
# Save model

training_model.save('e2fs2s.hdf5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
[decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)