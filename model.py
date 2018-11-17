import tensorflow as tf
import numpy as np


class Network:

    def __init__(self, config):
        self.config = config
        self.train_step = 0
        conf = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.initialize_network()
        self.sess.run(tf.initialize_all_variables())

    def initialize_network(self):
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.input_shape, name="image_input")
        out = self.image_input
        print(out)
        with tf.variable_scope("cnn_part"):
            for filters, kernel_size, strides in zip(self.config.filters, self.config.kernel_size, self.config.strides):
                layer = tf.layers.conv3d(
                    inputs=out,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    activation=tf.nn.relu,
                    padding=self.config.padding
                )
                print(layer)
                out = layer

        self.cnn_output = tf.layers.flatten(out)
        out = self.cnn_output

        with tf.variable_scope("dnn_part"):
            for output_num in self.config.dnn_shape:
                layer = tf.layers.dense(
                    inputs=out,
                    units=output_num,
                    activation=tf.nn.relu
                )
                out = layer

        self.dnn_output = out
        print(self.dnn_output)

        self.standard_mat = tf.placeholder(
            tf.float32, shape=[None, 16 * 7], name="standard_mat")
        with tf.variable_scope("train_part"):
            self.loss = tf.reduce_mean(
                (tf.square(self.dnn_output - self.standard_mat)))
            self.trainer = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(self.loss)

    def train(self, input_upside_buffer, input_downside_buffer, output_upside_buffer, output_downside_buffer):
        self.train_step = self.train_step + 1
        # print(type(input_upside_buffer))
        # print(type(output_upside_buffer))
        _, loss = self.sess.run([self.trainer, self.loss], feed_dict={
            self.image_input: input_upside_buffer,
            self.standard_mat: output_upside_buffer
        })
        if self.train_step % 10 == 1:
            print("now learning step: %d, now loss: %f" %
                  (self.train_step, loss))

    def test(self, data):
        loss, output = self.sess.run([self.loss, self.dnn_output], feed_dict={
            self.image_input: data["image_input"],
            self.standard_mat: data["standard_mat"]
        })
        print("loss on test set: %f, output:" % loss, output)

    def model_save(self):
        pass
