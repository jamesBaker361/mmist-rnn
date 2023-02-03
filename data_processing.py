# keras module for building LSTM 
from keras_preprocessing.sequence import pad_sequences
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

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from mnist_dijk import *

def get_sequence_of_tokens(corpus):
    all_input_sequences = []
    all_words=set()
    for token_list in corpus:
        all_words.add(token_list[0])
        all_words.add(token_list[-1])
        for t in range(1, len(token_list)-1):
            all_words.add(token_list[t])
            n_gram_sequence = token_list[:t+1]
            all_input_sequences.append(n_gram_sequence)

    return all_input_sequences, len(all_words)

def to_category(token,total_words): #token is number < total_words +1
    arr=np.zeros(total_words)
    arr[token]=1
    return arr


def generate_padded_sequences(input_sequences,total_words=28*28):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre', value=0))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = np.array([to_category(l,total_words) for l in label])
    return predictors, label, max_sequence_len