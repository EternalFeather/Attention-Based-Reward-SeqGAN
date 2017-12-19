# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np

class Rollout(object):
	def __init__(self, model, update_rate):
		self.model = model
		self.update_rate = update_rate

		self.num_embed = self.model.num_embed
		self.batch_size = self.model.batch_size
		self.embed_dim = self.model.embed_dim
		self.hidden_dim = self.model.hidden_dim
		self.sequence_length = self.model.sequence_length
		self.start_token = tf.identity(self.model.start_token)
		self.learning_rate = tf.identity(self.model.learning_rate)

		self.r_embeddings = tf.identity(self.model.g_embeddings)
		self.r_recurrent_unit = self.create_recurrent_unit()
		self.r_output_unit = self.create_output_unit()

		# placeholder
		self.x = tf.placeholder(tf.int32, shape = [self.batch_size, self.sequence_length])
		self.given_num = tf.placeholder(tf.int32)

		# process for batch
		self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.r_embeddings, self.x), perm = [1, 0, 2])

		ta_embed_x = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.sequence_length, \
													dynamic_size = False, infer_shape = True)
		ta_embed_x = ta_embed_x.unstack(self.processed_x)

		ta_x = tensor_array_ops.TensorArray(dtype = tf.int32, size = self.sequence_length, \
											dynamic_size = False, infer_shape = True)
		ta_x = ta_x.unstack(tf.transpose(self.x, perm = [1, 0]))

		self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
		self.h0 = tf.stack([self.h0, self.h0])

		gen_x = tensor_array_ops.TensorArray(dtype = tf.int32, size = self.sequence_length, \
											dynamic_size = False, infer_shape = True)

		# when current i < given_num, use the provided tokens as the input as each time step
		def _r_recurrence_1(i, x, h, given_num, gen_x):
			h_t = self.r_recurrent_unit(x, h)
			x_ = ta_embed_x.read(i)
			gen_x = gen_x.write(i, ta_x.read(i))
			return i + 1, x_, h_t, given_num, gen_x

		# when current i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
		def _r_recurrence_2(i, x, h, given_num, gen_x):
			h_t = self.r_recurrent_unit(x, h)
			o_t = self.r_output_unit(h_t)
			log_prob = tf.log(o_t)
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
			x_ = tf.nn.embedding_lookup(self.r_embeddings, next_token)
			gen_x = gen_x.write(i, next_token)
			return i + 1, x_, h_t, given_num, gen_x

		i, x_t, h_t, given_num, self.gen_x = control_flow_ops.while_loop(
			cond = lambda i, _1, _2, given_num, _4: i < given_num, \
			body = _r_recurrence_1, \
			loop_vars = (tf.constant(0, dtype = tf.int32), \
						tf.nn.embedding_lookup(self.r_embeddings, self.start_token), self.h0, self.given_num, gen_x))

		_, _, _, _, self.gen_x = control_flow_ops.while_loop(
			cond = lambda i, _1, _2, _3, _4: i < self.sequence_length, \
			body = _r_recurrence_2, \
			loop_vars = (i, x_t, h_t, given_num, self.gen_x))

		self.gen_x = self.gen_x.stack()
		self.gen_x = tf.transpose(self.gen_x, perm = [1, 0])


	def create_recurrent_unit(self):
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

		def unit(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(x, self.Wi) + \
				tf.matmul(hidden_state, self.Ui) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(x, self.Wf) + \
				tf.matmul(hidden_state, self.Uf) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(x, self.Wo) + \
				tf.matmul(hidden_state, self.Uo) + self.bo
			)

			c_ = tf.sigmoid(
				tf.matmul(x, self.Wc) + \
				tf.matmul(hidden_state, self.Uc) + self.bc
			)

			c = f * cell + c_ * i
			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return unit


	def create_output_unit(self):
		self.V = tf.identity(self.model.V)
		self.c = tf.identity(self.model.c)

		def unit(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return unit


	def get_reward(self, sess, input_x, rollout_num, discriminator):
		rewards = []
		for i in range(rollout_num):
			for given_num in range(1, 20):
				samples = sess.run(self.gen_x, feed_dict = {self.x: input_x, self.given_num: given_num})
				ypred_for_acu = sess.run(discriminator.y_pred_for_acu, feed_dict = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0})
				ypred = np.array([item[1] for item in ypred_for_acu])
				if i == 0:
					rewards.append(ypred)
				else:
					rewards[given_num - 1] += ypred
			# the last token reward
			ypred_for_acu = sess.run(discriminator.y_pred_for_acu, feed_dict = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0})
			ypred = np.array([item[1] for item in ypred_for_acu])
			if i == 0:
				rewards.append(ypred)
			else:
				rewards[19] += ypred

		rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)		# [batch_size * seq_len]
		return rewards


	def update_params(self):
		self.r_embeddings = tf.identity(self.model.g_embeddings)
		self.r_recurrent_unit = self.update_recurrent_unit()
		self.r_output_unit = self.update_output_unit()


	def update_recurrent_unit(self):
		self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.model.Wi)
		self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.model.Ui)
		self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.model.bi)

		self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.model.Wf)
		self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.model.Uf)
		self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.model.bf)

		self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.model.Wo)
		self.Uo = self.update_rate * self.Uo + (1 - self.update_rate) * tf.identity(self.model.Uo)
		self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.model.bo)

		self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.model.Wc)
		self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.model.Uc)
		self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.model.bc)

		def unit(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(x, self.Wi) + \
				tf.matmul(hidden_state, self.Ui) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(x, self.Wf) + \
				tf.matmul(hidden_state, self.Uf) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(x, self.Wo) + \
				tf.matmul(hidden_state, self.Uo) + self.bo
			)

			c_ = tf.sigmoid(
				tf.matmul(x, self.Wc) + \
				tf.matmul(hidden_state, self.Uc) + self.bc
			)

			c = f * cell + i * c_
			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return unit


	def update_output_unit(self):
		self.V = self.update_rate * self.V + (1 - self.update_rate) * tf.identity(self.model.V)
		self.c = self.update_rate * self.c + (1 - self.update_rate) * tf.identity(self.model.c)

		def unit(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return unit