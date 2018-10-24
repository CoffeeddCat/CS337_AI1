import tensorflow as tf
import numpy as np

class Network:
    def __init__(self, config):
        self.config = config
        self.session = tf.Session()
        self.image_input = tf.placeholder(tf.float32, shape=[None] + config.input_shape, name="image_input")
        with tf.variable_scope("cnn_part"):
            