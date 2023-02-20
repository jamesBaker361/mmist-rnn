# keras module for building LSTM 
from keras_preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import keras

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
        padded=np.array(pad_sequences([current_sequence], maxlen=max_sequence_len-1, padding='pre')).astype(int)
        #print(padded)
        token=np.argmax(model(padded))
        current_sequence.append(token)
    return current_sequence

def sequence_to_image(sequence,dim=28):
    empty_c=[[0.0 for _ in range(dim)] for __ in range(dim)]
    for token in sequence:
        (vert,horiz)=num_to_coords(token)
        empty_c[vert][horiz]=1
    return empty_c

def get_starting_points(dataset):
    labels={}
    for d in dataset:
        l=d["label"]
        if l not in labels:
            labels[l]={}
        start=d["sequence"][0]
        if start not in labels[l]:
            labels[l][start]=0
        labels[l][start]+=1
    for l in labels:
        labels[l]=dict(sorted(labels[l].items(), key=lambda item: -1*item[1]))
    return labels

def sequence_to_savedimg(sequence,name):
    plt.imshow(sequence_to_image(sequence))
    plt.savefig(name)

class GenImgCallback(keras.callbacks.Callback):
    def __init__(self, points,model,max_seq_len,path,drawings_per_class=1) -> None:
        super().__init__()
        self.points=points #{'class1':{point:frequency, point2: frequency2,,,,}} where frequency > frequency2
        self.model=model
        self.max_seq_len=max_seq_len
        self.path=path #path to save
        os.makedirs(path, exist_ok=True)
        self.drawings_per_class=drawings_per_class

    def on_epoch_end(self, epoch, logs=None):
        for class_name, points in self.points.items():
            for p in [_p for _p in points.keys()][:self.drawings_per_class]:
                sequence=generate_sequence([p],self.max_seq_len, self.model, self.max_seq_len)
                sequence_to_savedimg(sequence,'{}/{}_{}_{}.png'.format(self.path,class_name,epoch,p))
                print('{}/{}_{}_{}.png saved!'.format(self.path,class_name,epoch,p))

class GenImgCallbackAttention(GenImgCallback):
    def __init__(self, points, model, max_seq_len,path,length,drawings_per_class=1) -> None:
        super().__init__(points, model, max_seq_len,path,drawings_per_class=1)
        self.length=length #how long 

    def on_epoch_end(self, epoch, logs=None):
        for class_name, points in self.points.items():
            for p in [_p for _p in points.keys()][:self.drawings_per_class]:
                sequence=[]
                initial_input=np.asarray([[0 for _ in range(self.max_seq_len-1)]+[p]])
                for _ in range(self.length):
                    new_token=initial_input[0][-1]
                    sequence.append(new_token)
                    if new_token<1:
                        break
                    y=self.model(initial_input)[0]
                    initial_input=np.asarray([[np.argmax(t) for t in y]])
                sequence_to_savedimg(sequence,'{}/{}_{}_{}.png'.format(self.path,class_name,epoch,p))
                print('{}/{}_{}_{}.png saved!'.format(self.path,class_name,epoch,p))

class GenImgCallbackAttentionConditional(GenImgCallbackAttention):
    def __init__(self,oh_encoder, points, model, max_seq_len, path, length, drawings_per_class=1) -> None:
        super().__init__(points, model, max_seq_len, path, length, drawings_per_class)
        self.oh_encoder=oh_encoder

    def on_epoch_end(self, epoch, logs=None):
        for class_name, points in self.points.items():
            for p in [_p for _p in points.keys()][:self.drawings_per_class]:
                reshaped=np.reshape([class_name],(-1,1))
                encoded_labels=self.oh_encoder.transform(reshaped).toarray()
                expanded=tf.expand_dims(encoded_labels,-2)
                repeated=tf.repeat(expanded,self.max_seq_len,-2)
                sequence=[]
                initial_input=np.asarray([[0 for _ in range(self.max_seq_len-1)]+[p]])
                for _ in range(self.length):
                    new_token=initial_input[0][-1]
                    sequence.append(new_token)
                    if new_token<1:
                        break
                    y=self.model([initial_input,repeated])[0]
                    initial_input=np.asarray([[np.argmax(t) for t in y]])
                sequence_to_savedimg(sequence,'{}/{}_{}_{}.png'.format(self.path,class_name,epoch,p))
                print('{}/{}_{}_{}.png saved!'.format(self.path,class_name,epoch,p))