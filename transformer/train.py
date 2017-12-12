# -*- coding:utf-8 -*-
import tensorflow as tf
from transformer.data_loader import Data_helper
from transformer.params import Params as pm
from transformer.modules import Model


class Transformer(object):
	def __init__(self, trainable=True):
		self.data_loader = Data_helper()
		self.models = Model()

		if trainable:
			self.x, self.y, self.num_batch = self.data_loader.mini_batch()
		else:
			# Re-initialize
			self.x = tf.placeholder(tf.int32, shape=(None, pm.SEQ_LEN))
			self.y = tf.placeholder(tf.int32, shape=(None, pm.SEQ_LEN))

		# Add 2(<SOS>) in the beginning
		start_token = tf.ones([pm.BATCH_SIZE, 1], dtype=tf.int32) * 2
		self.decoder_inputs = tf.concat((start_token, self.y[:, :-1]), -1)

		# Load Vocabulary
		self.de2idx, self.idx2de = self.data_loader.load_vocab(pm.DECODER_VOCAB)
		self.en2idx, self.idx2en = self.data_loader.load_vocab(pm.ENCODER_VOCAB)

# Module -----------------

		# Encoder
		with tf.variable_scope("encoder"):
			# Input Embedding
			self.encoder = self.models.embedding(self.x,
												vocab_size=len(self.en2idx),
												num_units=pm.HIDDEN_UNITS,
												scale=True,
												scope="Input_Embedding")

			# Positional Encoding
			self.encoder += self.models.positional_encoding(self.x,
															num_units=pm.HIDDEN_UNITS,
															zero_pad=False,
															scale=False,
															scope="En_Positional_Encoding")

			# Dropout
			self.encoder = tf.layers.dropout(self.encoder,
											rate=pm.DROPOUT_RATE,
											training=tf.convert_to_tensor(trainable))

			# Body_networks
			for num in range(pm.ENCODER_N):
				with tf.variable_scope("encoder_networds_{}".format(num)):
					# Multi-Head Attention
					self.encoder = self.models.multihead_attention(queries=self.encoder,
																	keys=self.encoder,
																	num_units=pm.HIDDEN_UNITS,
																	num_heads=pm.NUM_HEADS,
																	dropout_rate=pm.DROPOUT_RATE,
																	is_training=trainable,
																	mask=False)

					# Feed Forward
					self.encoder = self.models.feedforward(self.encoder,
															num_units=(4 * pm.HIDDEN_UNITS, pm.HIDDEN_UNITS))

		# Decoder
		with tf.variable_scope("decoder"):
			# Output Embedding
			self.decoder = self.models.embedding(self.decoder_inputs,
												vocab_size=len(self.en2idx),
												num_units=pm.HIDDEN_UNITS,
												scale=True,
												scope="Output_embedding")

			# Positional Encoding
			self.decoder += self.models.positional_encoding(self.decoder_inputs,
															num_units=pm.HIDDEN_UNITS,
															zero_pad=False,
															scale=False,
															scope="De_Positional_Encoding")

			# Dropout
			self.decoder = tf.layers.dropout(self.decoder,
											rate=pm.DROPOUT_RATE,
											training=tf.convert_to_tensor(trainable))

			# Body_networks
			for num in range(pm.DECODER_N):
				with tf.variable_scope("decoder_networks_{}".format(num)):
					# Multi-Head Attention with Mask(self-attention)
					self.decoder = self.models.multihead_attention(queries=self.decoder,
																	keys=self.decoder,
																	num_units=pm.HIDDEN_UNITS,
																	num_heads=pm.NUM_HEADS,
																	dropout_rate=pm.DROPOUT_RATE,
																	is_training=trainable,
																	mask=True,
																	scope="De_Multihead_Attention")

					# Multi-Head Attention(vanilla attention)
					self.decoder = self.models.multihead_attention(queries=self.decoder,
																	keys=self.decoder,
																	num_units=pm.HIDDEN_UNITS,
																	num_heads=pm.NUM_HEADS,
																	dropout_rate=pm.DROPOUT_RATE,
																	is_training=trainable,
																	mask=False,
																	scope="De_Vanilla_Attention")

					# Feed Forward
					self.decoder = self.models.feedforward(self.decoder,
															num_units=[4 * pm.HIDDEN_UNITS, pm.HIDDEN_UNITS])

		# Linear & Softmax
		self.logits = tf.layers.dense(self.decoder, len(self.en2idx))
		self.predicts = tf.cast(tf.argmax(tf.nn.softmax(self.logits), -1), tf.int32)
		self.is_target = tf.cast(tf.not_equal(self.y, 0), tf.float32)   # Distinguish zero(Padding) or not
		self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predicts, self.y), tf.float32) * self.is_target) / (tf.reduce_sum(self.is_target))
		tf.summary.scalar("accuracy", self.accuracy)

# End Module ----------------

		# Compile
		if trainable:
			# Loss
			self.y_smoothed = self.models.label_smoothing(tf.one_hot(self.y, depth=len(self.en2idx)))
			self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
			self.mean_loss = tf.reduce_mean(self.loss * self.is_target) / (tf.reduce_sum(self.is_target))

			# Optimizer
			self.global_step = tf.Variable(0, name="global_step", trainable=False)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=pm.LEARNING_RATE, beta1=0.9, beta2=0.98, epsilon=1e-8).minimize(self.mean_loss, global_step=self.global_step)

			# Summary
			tf.summary.scalar("mean_loss", self.mean_loss)
			self.merged_summary = tf.summary.merge_all()
