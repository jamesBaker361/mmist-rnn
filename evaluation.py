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

def generate_sequence(initial_sequence, length, model, max_sequence_len):
    current_sequence= initial_sequence
    for _ in range(length):
        print(current_sequence)
        padded=np.array(pad_sequences([current_sequence], maxlen=max_sequence_len-1, padding='pre')).astype(int)
        token=np.argmax(model(padded))
        current_sequence.append(token)
    return current_sequence

def sequence_to_image(sequence,dim=28):
    empty_c=[[0.0 for _ in range(dim)] for __ in range(dim)]
    for token in sequence:
        (vert,horiz)=num_to_coords(token)
        empty_c[vert][horiz]=1
    return empty_c