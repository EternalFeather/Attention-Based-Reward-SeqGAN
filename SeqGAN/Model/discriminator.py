# -*- coding:utf-8 -*-


class Discriminator(object):
	def __init__(self, seq_length, num_classes, vocab_size, emb_size, filter_sizes, num_filters, l2_reg_lambda):
		self.seq_length = seq_length
		self.num_classes = num_classes
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.l2_reg_lambda = l2_reg_lambda


