# keras module for building LSTM 
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
from sklearn.preprocessing import OneHotEncoder

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


def get_filtered_dataset(dataset,label_set):
    return dataset.filter(lambda x: str(x["label"]) in label_set)

def get_balanced_dataset(dataset,limit,label_set):
    label_maxes={l:-1 for l in label_set}
    for d in dataset:
        if str(d["label"]) in label_maxes:
            label_maxes[str(d["label"])]=max(label_maxes[str(d["label"])], d["occurence"])
    limit=min([limit]+[l for l in label_maxes.values()])
    label_count={l:0 for l in label_set}
    def _is_in_set(x):
        if x not in label_set:
            return False
        label_count[x]+=1
        if label_count[x]>=limit:
            return False
        return True

    return dataset.filter(lambda x: _is_in_set(str(x["label"])))
    '''label_to_data={l:[] for l in label_set}
    for d in dataset:
        if str(d["label"]) in label_to_data:
            label_to_data[str(d["label"])].append(d)
    min_length=min([len(series) for series in label_to_data.values()])
    limit=min(limit,min_length)
    raw=[]
    for i in range(limit):
        for series in label_to_data.values():
            raw.append(series[i])
    return raw'''


def get_sequence_of_tokens(corpus):
    '''It takes a list of lists of tokens, and returns a list of lists of tokens, and the number of unique
    tokens
    
    Parameters
    ----------
    corpus
        The corpus is the list of all the sequences in the dataset =[[100,101,103,,,500,501],,,[200,201,,,450,451]]
    
    Returns
    -------
        all_input_sequences is a list of lists, where each list is a sequence of tokens.[[100],[100,101],[101,101,103]]
        that are all the subsequences of all the sequences in corpus
        len(all_words) is the number of unique words in the corpus.
    
    '''
    all_input_sequences = []
    all_words=set()
    for token_list in corpus:
        if token_list[-1]==-1:
            token_list=token_list[:-1]
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
    '''
    input_sequences = [[a], [a,b], [a,b,c], [a,b,c,d],,]

    predictors = [[0,0,,,,a], [0,0,,,a,b], [0,0,,a,b,c]]
    label =[[b], [c], [d]]
    '''
    max_sequence_len = max([len(x) for x in input_sequences])
    print(max_sequence_len)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre', value=0))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = np.array([to_category(l,total_words) for l in label])
    return predictors, label, max_sequence_len

def get_sequences_attention(sub_seq_len, corpus,labels=[],total_words=28*28):
    '''Given a corpus of sequences, return a list of sequences of length `sub_seq_len` and a list of
    one-hot encoded sequences of length `sub_seq_len`
    
    Parameters
    ----------
    sub_seq_len
        The length of the subsequence.
    corpus
        a list of lists of tokens. [[200,201,202,204,,,,600],,,[300,302,310,,,600,601]]
    labels
        [1,2,2,4,5] the digit/chars that each list of tokens corresponds to
    
    '''
    x=[] #[[a,b,c] ,[b,c,d]]
    y=[] # one hot encoded version  of [[b,c,d], [c,d,e]]
    subseq_labels=[] #[1,1,]
    max_sequence_len = max([len(x) for x in corpus])
    padding = [0 for _ in range(sub_seq_len-1)]
    for cter, token_list in enumerate(corpus):
        token_list = padding+token_list+[0]
        for t in range(len(token_list)-sub_seq_len):
            if len(labels)>0:
                subseq_labels.append(labels[cter])
            x.append(token_list[t:t+sub_seq_len])
            y_unencoded=token_list[t+1:1+t+sub_seq_len]
            y.append([to_category(unenc_token, total_words) for unenc_token in y_unencoded])
    return np.asarray(x),np.asarray(y), subseq_labels

def expand_labels(subseq_labels, max_seq_len, n_classes=0, oh_encoder=None):
    #[a,b,a,c,c,c,,,] --> (-1 ,n_classes) --> (-1 ,max_seq_len,n_classes)
    if n_classes <1:
        n_classes= len(set(subseq_labels))
    subseq_labels=np.reshape(subseq_labels,(-1,1))
    if oh_encoder is None:
        oh_encoder=OneHotEncoder().fit(subseq_labels)
    encoded_labels=oh_encoder.transform(subseq_labels).toarray()
    expanded=tf.expand_dims(encoded_labels,-2)
    repeated=tf.repeat(expanded,max_seq_len,-2)
    return tf.cast(repeated,dtype=tf.float32), oh_encoder


            


    