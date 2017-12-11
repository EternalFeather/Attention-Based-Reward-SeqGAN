# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class Model(object):
	def __init__(self):
		pass

	def embedding(self, inputs, vocab_size, num_units, zero_pad=True, scale=True, scope="Input_Embedding", reuse=None):
		with tf.variable_scope(scope, reuse=reuse):
			embeddings = tf.Variable(tf.random_uniform([vocab_size, num_units], -1.0, 1.0), name="look_up")

			if zero_pad:
				embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
			outputs = tf.nn.embedding_lookup(embeddings, inputs)

			if scale:
				outputs = outputs * tf.sqrt(num_units)

		return outputs

	def positional_encoding(self, inputs, num_units, zero_pad=True, scale=True, scope="Positional_Encoding", reuse=None):
		N, T = inputs.get_shape().as_list()
		with tf.variable_scope(scope, reuse=reuse):
			position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # shape = [batch_size, seq_length]

			# PE function for sin & cos embeddings
			position_encoder = np.array([[pos / np.power(10000, 2.0 * i / num_units) for i in range(num_units // 2)] for pos in range(T)])
			position_encoder[:, 0::2] = np.sin(position_encoder[:, 0::2])
			position_encoder[:, 1::2] = np.cos(position_encoder[:, 1::2])

			# Convert to tensor
			embeddings = tf.convert_to_tensor(position_encoder)

			if zero_pad:
				embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
			outputs = tf.nn.embedding_lookup(embeddings, position_idx)

			if scale:
				outputs = outputs * tf.sqrt(num_units)

		return outputs

	def multihead_attention(self, queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True, mask=False, scope="Multihead_Attention", reuse=None):
		with tf.variable_scope(scope, reuse=reuse):
			# Set the fall back option for num_units
			if num_units is None:
				num_units = queries.get_shape().as_list()[-1]

			# Linear
			Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # shape = [N, T, C]
			K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
			V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

			Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # shape = [h * N, T, C / h]
			K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
			V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

			# Attention(Q, K, V)
			outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))    # shape = [h * N, T_q, T_k]
			outputs = outputs / tf.sqrt(K_.get_shape().as_list()[-1])

			# Key Masking
			key_masks = tf.sign()


	def feedforward(self):
		pass

	def label_smoothing(self):
		pass
