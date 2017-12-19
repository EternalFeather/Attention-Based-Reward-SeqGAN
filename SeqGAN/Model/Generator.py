# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator(object):
	def __init__(self, num_embed, batch_size, embed_dim, hidden_dim, sequence_length, start_token, learning_rate, reward_gamma):
		self.num_embed = num_embed
		self.batch_size = batch_size
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.start_token = tf.constant([start_token] * self.batch_size, dtype = tf.int32)
		self.learning_rate = tf.Variable(float(learning_rate), trainable = False)
		self.reward_gamma = reward_gamma
		self.g_params = []
		self.grad_clip = 5.0
		self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

		with tf.variable_scope('generator'):
			self.g_embeddings = tf.Variable(self.init_matrix([self.num_embed, self.embed_dim]))
			self.g_params.append(self.g_embeddings)
			self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
			self.g_output_unit = self.create_output_unit(self.g_params)

		# placeholder definition
		self.x = tf.placeholder(tf.int32, shape = [self.batch_size, self.sequence_length])
		self.rewards = tf.placeholder(tf.float32, shape = [self.batch_size, self.sequence_length])

		# processed for batch
		self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm = [1, 0, 2])	# reshape to : [seq_len * batch_size * embed_dim]

		# Initial states
		self.h0 = tf.zeros([self.batch_size, self.hidden_dim])		# dim of hidden_state = [batch_size * hidden_dim]
		# ====== question -> 1 ====== (Solved : [hidden_state, cell])
		self.h0 = tf.stack([self.h0, self.h0])
		# ======(clear)

		gen_o = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length, \
											dynamic_size = False, infer_shape = True)	# with the static size and all elements are the same size
		gen_x = tensor_array_ops.TensorArray(dtype = tf.int32, size = self.sequence_length, \
											dynamic_size = False, infer_shape = True)

		# static function
		def _g_recurrence(i, x, h, gen_o, gen_x):
			h_t = self.g_recurrent_unit(x, h)
			o_t = self.g_output_unit(h_t)			# output_size = [batch_size * vocab_size] 
			log_prob = tf.log(o_t)
			# ====== question -> 2 ====== (Solved : Using different prob to choose token, next_token_shape = [batch_size])
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
			# ======(clear)
			x_ = tf.nn.embedding_lookup(self.g_embeddings, next_token)		# shape = [batch_size * embed_dim]
			# Prob
			gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_embed, 1.0, 0.0), \
												o_t), 1))		# one_hot_shape = [batch_size * num_embed(vocab_size)]
			gen_x = gen_x.write(i, next_token)
			return i + 1, x_, h_t, gen_o, gen_x


		def _pretrain_recurrence(i, x, h, g_predictions):
			h_t = self.g_recurrent_unit(x, h)
			o_t = self.g_output_unit(h_t)
			g_predictions = g_predictions.write(i, o_t)		# shape = [batch_size * vocab_size]
			x_ = ta_embed_x.read(i)
			return i + 1, x_, h_t, g_predictions


		_, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
				cond = lambda i, _1, _2, _3, _4: i < self.sequence_length, \
				body = _g_recurrence, \
				loop_vars = (tf.constant(0, dtype = tf.int32), \
							tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))


		self.gen_x = self.gen_x.stack()		# shape = [seq_len * batch_size]
		self.gen_x = tf.transpose(self.gen_x, perm = [1, 0])	# shape = [batch_size * seq_len]

		g_predictions = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length, \
													dynamic_size = False, infer_shape = True)

		ta_embed_x = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length, \
													dynamic_size = False, infer_shape = True)
		ta_embed_x = ta_embed_x.unstack(self.processed_x)
		
		# Pre-train for generator
		_, _, _, self.g_predictions = control_flow_ops.while_loop(
				cond = lambda i, _1, _2, _3: i < self.sequence_length, \
				body = _pretrain_recurrence, \
				loop_vars = (tf.constant(0, dtype = tf.int32), \
							tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, g_predictions))

		self.g_predictions = self.g_predictions.stack()		# shape = [seq_len * batch_size * vocab_size]
		self.g_predictions = tf.transpose(self.g_predictions, perm = [1, 0, 2])	# shape = [batch_size * seq_len * vocab_size]

		# Pre-train loss
		self.pretrain_loss = -tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_embed, 1.0, 0.0) * \
											tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_embed]), 1e-20, 1.0))) / \
											(self.sequence_length * self.batch_size)			# shape_of_one_hot = [seq_len * batch_size, vocab_size]
		
		# training updates
		pretrain_opt = tf.train.AdamOptimizer(self.learning_rate)

		self.pretrain_grad, norm = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
		self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

		# Unsupervised Training
		self.g_loss = -tf.reduce_sum(tf.reduce_sum(tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.num_embed, 1.0, 0.0) * \
									tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_embed]), 1e-20, 1.0)), 1) * tf.reshape(self.rewards, [-1]))

		g_opt = tf.train.AdamOptimizer(self.learning_rate)

		self.g_grad, norm = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
		self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))


	def init_matrix(self, shape):
		return tf.random_normal(shape, stddev=0.1)


	def create_recurrent_unit(self, params):
		self.Wi = tf.Variable(self.init_matrix([self.embed_dim, self.hidden_dim]))
		self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
		self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

		self.Wf = tf.Variable(self.init_matrix([self.embed_dim, self.hidden_dim]))
		self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
		self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

		self.Wo = tf.Variable(self.init_matrix([self.embed_dim, self.hidden_dim]))
		self.Uo = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
		self.bo = tf.Variable(self.init_matrix([self.hidden_dim]))

		self.Wc = tf.Variable(self.init_matrix([self.embed_dim, self.hidden_dim]))
		self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
		self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))

		params.extend([
			self.Wi, self.Ui, self.bi,
			self.Wf, self.Uf, self.bf,
			self.Wo, self.Uo, self.bo,
			self.Wc, self.Uc, self.bc])

		def unit(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(x, self.Wi) +
				tf.matmul(hidden_state, self.Ui) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(x, self.Wf) +
				tf.matmul(hidden_state, self.Uf) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(x, self.Wo) +
				tf.matmul(hidden_state, self.Uo) + self.bo
			)

			c_ = tf.nn.tanh(
				tf.matmul(x, self.Wc) +
				tf.matmul(hidden_state, self.Uc) + self.bc
			)

			c = f * cell + i * c_

			current_hidden_state = o * tf.nn.tanh(c)
			return tf.stack([current_hidden_state, c])

		return unit


	def create_output_unit(self, params):
		self.V = tf.Variable(self.init_matrix([self.hidden_dim, self.num_embed]))	# one-hot
		self.c = tf.Variable(self.init_matrix([self.num_embed]))

		params.extend([self.V, self.c])

		def unit(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)		# normalization
			return output

		return unit


	def g_optimizer(self, *args, **kwargs):
		return tf.train.AdamOptimizer(*args, **kwargs)


	def pretrain_step(self, sess, x):
		outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict = {self.x: x})
		return outputs


	def generate(self, sess):
		outputs = sess.run(self.gen_x)
		return outputs
