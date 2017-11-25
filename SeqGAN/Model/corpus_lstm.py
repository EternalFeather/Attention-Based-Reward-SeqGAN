# -*- coding:utf-8 -*-


class Corpus_lstm(object):
	def __init__(self, vocab_size, emb_size, hidden_size, seq_length, start_token, params):
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.seq_length = seq_length
		self.start_token = start_token
		self.params = params

	def generate(self):
		pass

	def create_recurrent_unit(self):
		pass

	def create_output_unit(self):
		pass
