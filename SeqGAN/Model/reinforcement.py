# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from SeqGAN.Config.hyperparameters import Hyperparameter as pm


class Reinforcement(object):
	def __init__(self, model, update_rate):
		self.model = model
		self.update_rate = update_rate

		self.vocab_size = self.model.vocab_size
		self.batch_size = self.model.batch_size
		self.emb_size = self.model.emb_size
		self.hidden_size = self.model.hidden_size
		self.seq_length = self.model.seq_length
		self.start_token = tf.identity(self.model.start_token)
		self.learning_rate = tf.identity(self.model.learning_rate)

		self.rl_embeddings = tf.identity(self.model.g_embeddings)
		self.rl_lstm_forward = self.lstm_forward()
		self.rl_linear_forward = self.linear_forward()

		# Placeholder
		self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_length])
		self.given_num = tf.placeholder(tf.int32)

		# Processed for batch(Real data)
		self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.rl_embeddings, self.x), perm=[1, 0, 2])

		ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		ta_emb_x = ta_emb_x.unstack(self.processed_x)

		ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))

		self.h0 = tf.zeros([self.batch_size, self.hidden_size])
		self.h0 = tf.stack([self.h0, self.h0])

		token_sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_length, dynamic_size=False, infer_shape=True)

		# When current index i < given_num, use the provided tokens as the input at each time step
		def _rl_recurrence_1(i, x_t, h_tm, given_num, gen_x):
			h_t = self.rl_lstm_forward(x_t, h_tm)
			x_ = ta_emb_x.read(i)
			gen_x = gen_x.write(i, ta_x.read(i))
			return i + 1, x_, h_t, given_num, gen_x

		# When current index >= given_num, start roll-out, use the output at time step t as the input at time step t+1
		def _rl_recurrence_2(i, x_t, h_tm, given_num, gen_x):
			h_t = self.rl_lstm_forward(x_t, h_tm)
			o_t = self.rl_linear_forward(h_t)
			log_prob = tf.log(o_t)
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
			x_ = tf.nn.embedding_lookup(self.rl_embeddings, next_token)
			gen_x = gen_x.write(i, next_token)
			return i + 1, x_, h_t, given_num, gen_x

		i, x_t, h_tm, given_num, self.token_sequence = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, given_num, _4: i < given_num,
			body=_rl_recurrence_1,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.rl_embeddings, self.start_token),
						self.h0, self.given_num, token_sequence)
		)

		_, _, _, _, self.token_sequence = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3, _4: i < self.seq_length,
			body=_rl_recurrence_2,
			loop_vars=(i, x_t, h_tm, given_num, self.token_sequence)
		)

		self.token_sequence = self.token_sequence.stack()
		self.token_sequence = tf.transpose(self.token_sequence, perm=[1, 0])    # shape = [batch_size, seq_length]

	def lstm_forward(self):
		self.Wi = tf.identity(self.model.Wi)
		self.Ui = tf.identity(self.model.Ui)
		self.bi = tf.identity(self.model.bi)

		self.Wf = tf.identity(self.model.Wf)
		self.Uf = tf.identity(self.model.Uf)
		self.bf = tf.identity(self.model.bf)

		self.Wo = tf.identity(self.model.Wo)
		self.Uo = tf.identity(self.model.Uo)
		self.bo = tf.identity(self.model.bo)

		self.Wc = tf.identity(self.model.Wc)
		self.Uc = tf.identity(self.model.Uc)
		self.bc = tf.identity(self.model.bc)

		def forward(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(x, self.Wi) + tf.matmul(hidden_state, self.Ui) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(x, self.Wf) + tf.matmul(hidden_state, self.Uf) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(x, self.Wo) + tf.matmul(hidden_state, self.Uo) + self.bo
			)

			c_ = tf.nn.tanh(
				tf.matmul(x, self.Wc) + tf.matmul(hidden_state, self.Uc) + self.bc
			)

			c = f * cell + i * c_

			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return forward

	def linear_forward(self):
		self.V = tf.identity(self.model.V)
		self.c = tf.identity(self.model.c)

		def forward(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return forward

	def get_reward(self, sess, x, rollout_num, discriminator):
		rewards = []
		for i in range(rollout_num):
			for given_num in range(1, 20):
				samples = sess.run(self.token_sequence, feed_dict={self.x: x, self.given_num: given_num})
				ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.x: samples, discriminator.dropout_keep_prob: pm.ADVERSARIAL_DROPOUT})
				ypred = np.array([item[1] for item in ypred_for_auc])


