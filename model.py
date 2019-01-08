import tensorflow as tf
import numpy as np


class Network:

    def __init__(self, config, scope):
        self.config = config
        self.train_step = 0
        self.scope = scope
        conf = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.initialize_network()
        self.saver = tf.train.Saver(max_to_keep=3)
        if self.config.model_load == True:
            self.model_load()
        else:
            self.sess.run(tf.initialize_all_variables())
        # self.saver = tf.train.Saver(max_to_keep=3)

    def initialize_network(self):
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.input_shape, name="image_input" + self.scope)
        out = self.image_input
        print(out)
        with tf.variable_scope("cnn_part" + self.scope):
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

        with tf.variable_scope("dnn_part" + self.scope):
            for output_num in self.config.dnn_shape:
                if output_num != 16*7:
                  fn = tf.nn.relu
                else:
                  fn = None

                layer = tf.layers.dense(
                    inputs=out,
                    units=output_num,
                    activation=fn
                )
                out = layer

        self.dnn_output = out
        print(self.dnn_output)

        self.standard_mat = tf.placeholder(
            tf.float32, shape=[None, 16 * 7], name="standard_mat" + self.scope)
        with tf.variable_scope("train_part" + self.scope):
            self.loss = tf.reduce_mean(
                (tf.square(self.dnn_output - self.standard_mat)))
            self.trainer = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(self.loss)

    def train(self, input_buffer, output_buffer):
        self.train_step = self.train_step + 1
        # print(type(input_upside_buffer))
        # print(type(output_upside_buffer))
        _, loss = self.sess.run([self.trainer, self.loss], feed_dict={
            self.image_input: input_buffer,
            self.standard_mat: output_buffer
        })
        if self.train_step % 10 == 1:
            print(self.scope + "now learning step: %d, now loss: %f" %
                  (self.train_step, loss))

        if self.train_step % self.config.every_steps_save == 1:
            self.model_save()

    def test(self, data):
        loss, output = self.sess.run([self.loss, self.dnn_output], feed_dict={
            self.image_input: data["image_input"],
            self.standard_mat: data["standard_mat"]
        })
        print("loss on test set:", loss)
        print("output:", output)

    def return_mat(self, data):
        output = self.sess.run([self.dnn_output], feed_dict={
            self.image_input: data
        })
        return output

    def model_save(self, name=None):
        print("now training step %d...model saving..." % (self.train_step))
        if name == None:
            self.saver.save(self.sess, "model/training_step" + self.scope,
                            global_step=self.train_step)
        else:
            self.saver.save(self.sess, name)

    def model_load(self):
        self.saver.restore(self.sess, "model/training_step" + self.scope + "_26001")
        print(self.scope, "load over.")
