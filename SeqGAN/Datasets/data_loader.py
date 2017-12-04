# -*- coding:utf-8 -*-
import numpy as np
import codecs


class Gen_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])

	def mini_batches(self, data_path):
		token_seqs = []
		with codecs.open(data_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')



class Dis_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])
		self.labels = np.array([])
