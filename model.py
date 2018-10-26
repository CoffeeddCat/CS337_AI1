import tensorflow as tf
import numpy as np

class Network:
    def __init__(self, config):
        self.config = config
        self.session = tf.Session()
        self.image_input = tf.placeholder(tf.float32, shape=[None] + config.input_shape, name="image_input")
        out = self.image_input
        with tf.variable_scope("cnn_part"):
            for filters, kernel_size, strides in zip(config.filters, config.kernel_size, config.strides):
                layer = tf.layers.conv3d(
                    inputs=out,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    activation=tf.nn.relu
                    )
                out = layer

        self.cnn_output = tf.layers.flatten(out)
        out = self.cnn_output

        with tf.variable_scope("dnn_part"):
            for output_num in config.dnn_shape:
                layer = tf.layers.dense(
                    inputs=out,
                    units=output_num,
                    activation=tf.nn.relu
                    )
                out = layer

        self.dnn_output = out

        