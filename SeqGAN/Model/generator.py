# -*- coding:utf-8 -*-


class Generator(object):
	def __init__(self, vocab_size, batch_size, emb_size, hidden_size, seq_length, start_token, learning_rate, reward_gamma):
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.seq_length = seq_length
		self.start_token = start_token
		self.learning_rate = learning_rate
		self.reward_gamma = reward_gamma



