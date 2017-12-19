# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Target_lstm(object):
	def __init__(self, num_embed, batch_size, embed_dim, hidden_dim, sequence_length, start_token, params):
		self.num_embed = num_embed
		self.batch_size = batch_size
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.start_token = tf.constant([start_token] * self.batch_size, dtype = tf.int32)
		self.o_params = []
		self.temperature = 1.0
		self.params = params
		tf.set_random_seed(66)

		with tf.variable_scope('target_lstm'):
			self.o_embeddings = tf.Variable(self.params[0])
			self.o_params.append(self.o_embeddings)
			self.o_recurrent_unit = self.create_recurrent_unit(self.o_params)
			self.o_output_unit = self.create_output_unit(self.o_params)

		self.x = tf.placeholder(tf.int32, shape = [self.batch_size, self.sequence_length])

		self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.o_embeddings, self.x), perm = [1, 0, 2])

		self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
		self.h0 = tf.stack([self.h0, self.h0])

		lstm_o = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length, \
												dynamic_size = False, infer_shape = True)
		lstm_x = tensor_array_ops.TensorArray(dtype = tf.int32, size = self.sequence_length, \
												dynamic_size = False, infer_shape = True)

		def _g_recurrence(i, x, h, lstm_o, lstm_x):
			h_t = self.o_recurrent_unit(x, h)
			o_t = self.o_output_unit(h_t)
			log_prob = tf.log(o_t)
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
			x_ = tf.nn.embedding_lookup(self.o_embeddings, next_token)
			lstm_o = lstm_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_embed, 1.0, 0.0), \
													o_t), 1))
			lstm_x = lstm_x.write(i, next_token)
			return i + 1, x_, h_t, lstm_o, lstm_x

		def _pretrain_recurrence(i, x, h, o_predictions, ta_emb_x):
			h_t = self.o_recurrent_unit(x, h)
			o_t = self.o_output_unit(h_t)
			o_predictions = o_predictions.write(i, o_t)
			x_ = ta_emb_x.read(i)
			return i + 1, x_, h_t, o_predictions, ta_emb_x


		_, _, _, self.lstm_o, self.lstm_x = control_flow_ops.while_loop(
			cond = lambda i, _1, _2, _3, _4: i < self.sequence_length, \
			body = _g_recurrence, \
			loop_vars = (tf.constant(0, dtype = tf.int32), \
						tf.nn.embedding_lookup(self.o_embeddings, self.start_token), self.h0, lstm_o, lstm_x))

		self.lstm_x = self.lstm_x.stack()
		self.lstm_x = tf.transpose(self.lstm_x, perm = [1, 0])

		# supervised pretraining for lstm
		o_predictions = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length, \
													dynamic_size = False, infer_shape = True)

		ta_emb_x = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length)
		ta_emb_x = ta_emb_x.unstack(self.processed_x)

		_, _, _, self.o_predictions, self.ta_emb_x = control_flow_ops.while_loop(
			cond = lambda i, _1, _2, _3, _4: i < self.sequence_length, \
			body = _pretrain_recurrence, \
			loop_vars = (tf.constant(0, dtype = tf.int32), \
						tf.nn.embedding_lookup(self.o_embeddings, self.start_token), self.h0, o_predictions, ta_emb_x))

		self.o_predictions = self.o_predictions.stack()
		self.o_predictions = tf.transpose(self.o_predictions, perm = [1, 0, 2])

		self.pretrain_loss = -tf.reduce_sum(tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.num_embed, 1.0, 0.0) * \
											tf.log(tf.reshape(self.o_predictions, [-1, self.num_embed]))) / (self.sequence_length * self.batch_size)

		self.out_loss = tf.reduce_sum(
			tf.reshape(-tf.reduce_sum(tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.num_embed, 1.0, 0.0) * \
									tf.log(tf.reshape(self.o_predictions, [-1, self.num_embed])), 1), [-1, self.sequence_length]), 1)


	def create_recurrent_unit(self, params):
		self.Wi = tf.Variable(self.params[1])
		self.Ui = tf.Variable(self.params[2])
		self.bi = tf.Variable(self.params[3])

		self.Wf = tf.Variable(self.params[4])
		self.Uf = tf.Variable(self.params[5])
		self.bf = tf.Variable(self.params[6])

		self.Wo = tf.Variable(self.params[7])
		self.Uo = tf.Variable(self.params[8])
		self.bo = tf.Variable(self.params[9])

		self.Wc = tf.Variable(self.params[10])
		self.Uc = tf.Variable(self.params[11])
		self.bc = tf.Variable(self.params[12])

		params.extend([
			self.Wi, self.Ui, self.bi,
			self.Wf, self.Uf, self.bf,
			self.Wo, self.Uo, self.bo,
			self.Wc, self.Uc, self.bc])

		def unit(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
					tf.matmul(x, self.Wi) + \
					tf.matmul(hidden_state, self.Ui) + self.bi \
				)

			f = tf.sigmoid(
					tf.matmul(x, self.Wf) + \
					tf.matmul(hidden_state, self.Uf) + self.bf \
				)

			o = tf.sigmoid(
					tf.matmul(x, self.Wo) + \
					tf.matmul(hidden_state, self.Uo) + self.bo \
				)

			c_ = tf.sigmoid(
					tf.matmul(x, self.Wc) + \
					tf.matmul(hidden_state, self.Uc) + self.bc \
				)

			c = f * cell + i * c_
			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return unit


	def create_output_unit(self, params):
		self.V = tf.Variable(self.params[13])
		self.c = tf.Variable(self.params[14])
		params.extend([self.Wo, self.bo])

		def unit(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return unit

	def generate(self, sess):
		outputs = sess.run(self.lstm_x)
		return outputs
