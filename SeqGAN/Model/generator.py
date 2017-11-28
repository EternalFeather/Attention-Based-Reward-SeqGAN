# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
	def __init__(self, vocab_size, batch_size, emb_size, hidden_size, seq_length, start_token, learning_rate, reward_gamma):
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.seq_length = seq_length
		self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
		self.learning_rate = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)
		self.reward_gamma = reward_gamma
		self.g_params = []
		self.grad_clip = 5.0
		self.expected_reward = tf.Variable(tf.zeros([self.seq_length]))

		with tf.variable_scope('generator'):
			self.g_embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.emb_size]))
			self.g_params.append(self.g_embeddings)     # shape = [1, vocab_size, emb_size]
			self.g_lstm_forward = self.recurrent_lstm_forward(self.g_params)
			self.g_linear_forward = self.recurrent_linear_forward(self.g_params)
# Initialize parameters ------------------

		# placeholder
		self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_length])
		# rewards shape[1] = self.seq_length comes from Monte-Carlo Search
		self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length])

		# processed for batch(Real datasets)
		self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # shape=[seq_length, batch_size, emb_size]

		# Init hidden state
		self.h0 = tf.zeros([self.batch_size, self.hidden_size])
		self.h0 = tf.stack([self.h0, self.h0])  # hidden_state + cell

		# input sequence is an array of tokens while output sequence is an array of probabilities
		output_prob_sequence = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		token_sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_length, dynamic_size=False, infer_shape=True)

# End initialize ------------------

# Forward step -------------------

		def _g_recurrence(i, x_t, h_tm, gen_o, gen_x):
			h_t = self.g_lstm_forward(x_t, h_tm)    # hidden_memory
			o_t = self.g_linear_forward(h_t)    # output of prob, shape = [batch_size, vocab_size]
			log_prob = tf.log(o_t)
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)  # using this softmax distribution to choose one token as next
			x_ = tf.nn.embedding_lookup(self.g_embeddings, next_token)
			gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0), o_t), 1))
			gen_x = gen_x.write(i, next_token)  # output_array, shape = [index_num(seq_length), batch_size]
			return i + 1, x_, h_t, gen_o, gen_x

		_, _, _, self.output_prob_sequence, self.input_sequence = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3, _4: i < self.seq_length,
			body=_g_recurrence,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
						self.h0, output_prob_sequence, token_sequence)
		)

		self.token_sequence = self.token_sequence.stack()   # shape = [seq_length, batch_size]
		self.token_sequence = tf.transpose(self.token_sequence, perm=[1, 0])    # shape = [batch_size, seq_length]

# End Forward step ----------------------

# Pre-train step -------------------------

		# Supervised pre-training for generator
		g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)

		# Real-data result of sequence
		ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		ta_emb_x = ta_emb_x.unstack(self.processed_x)

		def _pretrain_recurrence(i, x_t, h_tm, g_predictions):
			h_t = self.g_lstm_forward(x_t, h_tm)
			o_t = self.g_linear_forward(h_t)
			g_predictions = g_predictions.write(i, o_t)     # softmax_distribution, shape = [batch_size, vocab_size]
			x_ = ta_emb_x.read(i)   # read the next_token from real datasets with index i
			return i + 1, x_, h_t, g_predictions

		_, _, _, self.g_predictions = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3: i < self.seq_length,
			body=_pretrain_recurrence,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
						self.h0, g_predictions)
		)

		self.g_predictions = self.g_predictions.stack()
		# g_predictions is an softmax distribution array
		self.g_predictions = tf.transpose(self.g_predictions, perm=[1, 0, 2])    # shape = [batch_size, seq_length, vocab_size]

# End pre-train step -------------------

# Pre-train Compile configuration ----------------

		# pre-train_loss
		self.loss = -tf.reduce_sum(
			tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
				tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
			)
		) / (self.seq_length * self.batch_size)      # one_hot shape = [seq_length * batch_size, vocab_size]

		# pre-train_backward
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

# End Setting Pre-train Compile ------------------

# Training pre-train model ------------------

		self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.g_params), self.grad_clip)  # sum(list(g_params / loss))
		self.pretrain_updates = self.optimizer.apply_gradients(zip(self.pretrain_grad, self.g_params))

# End training pre-train --------------------

# Adversarial Learning train ----------------

		# output is a number represents the whole reward of sequence_tokens
		self.g_loss = -tf.reduce_sum(
			tf.reduce_sum(
				tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
					tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
				), 1) * tf.reshape(self.rewards, [-1])
		)   # Adversarial Learning with rewards, [seq_length * batch_size] * (sum of prob)
	# ==> sum([seq_length * batch_size] * (sum of prob) * (rewards))

		self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.g_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
		self.g_updates = self.g_optimizer.apply_gradients(zip(self.g_gradients, self.g_params))

# End Adversarial Learning ------------------

	def recurrent_lstm_forward(self, params):
		self.Wi = tf.Variable(tf.random_normal([self.emb_size, self.hidden_size]))
		self.Ui = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size]))
		self.bi = tf.Variable(tf.random_normal([self.hidden_size]))

		self.Wf = tf.Variable(tf.random_normal([self.emb_size, self.hidden_size]))
		self.Uf = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size]))
		self.bf = tf.Variable(tf.random_normal([self.hidden_size]))

		self.Wo = tf.Variable(tf.random_normal([self.emb_size, self.hidden_size]))
		self.Uo = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size]))
		self.bo = tf.Variable(tf.random_normal([self.hidden_size]))

		self.Wc = tf.Variable(tf.random_normal([self.emb_size, self.hidden_size]))
		self.Uc = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size]))
		self.bc = tf.Variable(tf.random_normal([self.hidden_size]))
		params.extend([
			self.Wi, self.Ui, self.bi,
			self.Wf, self.Uf, self.bf,
			self.Wo, self.Uo, self.bo,
			self.Wc, self.Uc, self.bc
		])

		def forward(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(self.Wi, x) + tf.matmul(self.Ui, hidden_state) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(self.Wf, x) + tf.matmul(self.Uf, hidden_state) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(self.Wo, x) + tf.matmul(self.Uo, hidden_state) + self.bo
			)

			c_ = tf.nn.tanh(
				tf.matmul(self.Wc, x) + tf.matmul(self.Uc, hidden_state) + self.bc
			)

			c = f * cell + i * c_

			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return forward

	# output shape = [batch_size, vocab_size] represents which word we should choose next time
	def recurrent_linear_forward(self, params):
		self.V = tf.Variable(tf.random_normal([self.hidden_size, self.vocab_size]))
		self.c = tf.Variable(tf.random_normal([self.vocab_size]))
		params.extend([
			self.V, self.c
		])

		def forward(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return forward









