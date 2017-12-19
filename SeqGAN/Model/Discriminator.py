# -*- coding:utf-8 -*-

import tensorflow as tf


def linear(input_, output_size, scope=None):
    """
    output[k] = sum_i(Matrix[k, i] * input_[i]) + Bias[k]
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix, [1, 0])) + bias_term


def highway(input_, size, num_layers, bias):
    """
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
    """
    with tf.variable_scope('Highway'):
        for idx in range(num_layers):
            g = tf.nn.relu(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(object):
    def __init__(self, sequence_length, num_classes, num_embed, embed_size, filter_sizes, hidden_sizes, learning_rate,
                 l2_reg_lambda):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_embed = num_embed
        self.embed_size = embed_size
        self.filter_sizes = filter_sizes
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Optimizer (Keep track of l2 regularization loss)
        l2_loss = tf.constant(0.0, dtype=tf.float32)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.name_scope('embedding'):
                self.W = tf.Variable(tf.random_uniform([self.num_embed, self.embed_size], -1.0, 1.0), name='W')
                self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)  # shape = [embed_size + 1]

            # create a convolution + max-pooling layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.hidden_sizes):
                with tf.name_scope('convolution-max-pooling-%s' % filter_size):
                    # Convolution layer
                    filter_shape = [filter_size, self.embed_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name='b')
                    conv = tf.nn.conv2d(self.embedding_chars_expanded, W, strides=[1, 1, 1, 1], \
                                        padding='VALID', name='conv')
                    h = tf.nn.relu(conv + b, name='relu')
                    # Pooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], \
                                            strides=[1, 1, 1, 1], padding='VALID', name='pool')
                    pooled_outputs.append(pooled)

            # Flatten
            self.hidden_sizes_total = sum(self.hidden_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flatten = tf.reshape(self.h_pool, [-1, self.hidden_sizes_total])

            # Add highway
            with tf.name_scope('highway'):
                self.h_highway = highway(self.h_pool_flatten, self.h_pool_flatten.get_shape()[1], 1, 0)
            # Add dropout
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # final scores and predictions
            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal([self.hidden_sizes_total, self.num_classes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                # self.scores = tf.nn.matmul(self.h_drop, W) + b
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.y_pred_for_acu = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

            # calculateMean loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
