from keras import layers, Model
import tensorflow as tf

class RepeatLayer(layers.Layer):
    def __init__(self,reps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reps=reps

    def call(self, x): #assuming tensor = (None, dim) returns (None, reps, dim)
        expanded=tf.expand_dims(x,-2)
        repeated=tf.repeat(expanded,self.reps,-2)
        return tf.cast(repeated,dtype=tf.float32)