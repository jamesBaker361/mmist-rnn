# keras module for building LSTM 
from keras_preprocessing.sequence import pad_sequences
from keras import layers, Model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 

# set seeds for reproducability
import tensorflow as tf
from numpy.random import seed
tf.random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os
import sys 
sys.path.append('./components/')
from miniGPT import TransformerBlock,TokenAndPositionEmbedding
from repeat import RepeatLayer

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def lstm_model(max_sequence_len, total_words=28*28,units=100, embedding_dim=10,n_layers=2,unit_multiplier=1):
    input_len = max_sequence_len 
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, embedding_dim, input_length=input_len))
    
    # Add Hidden Layer - LSTM Layer
    for _ in range(n_layers-1):
        model.add(LSTM(units,recurrent_dropout=0.2,return_sequences=True))
        units = max(4,int(units*unit_multiplier))
    model.add(LSTM(units,recurrent_dropout=0.2,return_sequences=False))
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def mini_gpt_model(max_sequence_len, total_words=28*28,units=100, embedding_dim=10,n_layers=2,unit_multiplier=1, num_heads=4):
    #inputs = [a,b,c,d] outputs=[b,c,d,e], but encoded as one-hot vectors
    inputs = layers.Input(shape=(max_sequence_len,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(max_sequence_len, total_words, embedding_dim)
    x = embedding_layer(inputs)
    for _ in range(n_layers-1):
        x = TransformerBlock(embedding_dim, num_heads, units)(x)
        units = max(4,int(units*unit_multiplier))
    x = TransformerBlock(embedding_dim, num_heads, units)(x)
    x = layers.Dense(total_words, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)

def mini_gpt_model_conditional(max_sequence_len, total_words=28*28,units=100, embedding_dim=13,n_layers=2,unit_multiplier=1, num_heads=4, n_classes=7,n_layers_class=3,units_class=32,activation_class='linear',condition_stage='both'):
    #inputs = [a,b,c,d] outputs=[b,c,d,e], but encoded as one-hot vectors
    if condition_stage not in set(['both', 'early', 'late']):
        #print('condition_stage set to both')
        condition_stage='both'
    inputs = layers.Input(shape=(max_sequence_len,), dtype=tf.int32)
    conditional_input=layers.Input(shape=(max_sequence_len, n_classes,))
    #cond_x=RepeatLayer(max_sequence_len)(conditional_input)
    cond_x=conditional_input
    for _ in range(n_layers_class):
        cond_x=layers.Dense(units_class, activation=activation_class)(cond_x)
    #conditional_input= tf.repeat(conditional_input, max_sequence_len, axis=-2)
    embedding_layer = TokenAndPositionEmbedding(max_sequence_len, total_words, embedding_dim)
    x = embedding_layer(inputs)
    if condition_stage == 'both' or condition_stage == 'early':
        x= tf.keras.layers.Concatenate(axis=-1)([x,cond_x])
        embedding_dim+=cond_x.shape[-1]
    for _ in range(n_layers-1):
        x = TransformerBlock(embedding_dim, num_heads, units)(x)
        units = max(4,int(units*unit_multiplier))
    x = TransformerBlock(embedding_dim, num_heads, units)(x)
    if condition_stage == 'both' or condition_stage == 'late':
        x= tf.keras.layers.Concatenate(axis=-1)([x,cond_x])
    x = layers.Dense(total_words, activation='softmax')(x)
    return Model(inputs=[inputs, conditional_input], outputs=x)