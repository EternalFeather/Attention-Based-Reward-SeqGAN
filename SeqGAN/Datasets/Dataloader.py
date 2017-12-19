# -*- coding:utf-8 -*-

import numpy as np
import codecs
from SeqGAN.Config.Hyperparameters import Parameters as pm


class Gen_dataloader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_stream = []

	def create_batch(self, data_file):
		self.token_stream = []
		with codecs.open(data_file, 'r', 'utf-8') as f:
			for line in f:
				line = line.strip()
				parse_line = list(map(int, line.split()))
				if len(parse_line) == pm.G_SEQ_LENGTH:
					self.token_stream.append(parse_line)

		# reduce some residual that out of batch
		self.num_batch = int(len(self.token_stream) / self.batch_size)
		self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
		self.token_stream = np.array(self.token_stream)
		self.sequence_batch = np.split(self.token_stream, self.num_batch, 0)
		self.pointer = 0

	def next_batch(self):
		temp = self.sequence_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch
		return temp

	def reset_pointer(self):
		self.pointer = 0


class Dis_dataloader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.sentence, self.labels = np.array([]), np.array([])

	def create_batch(self, positive_file, negative_file):
		positive_example, negative_example = [], []
		with codecs.open(positive_file, 'r', 'utf-8') as fp:
			for line in fp:
				line = line.strip()
				parse_line = list(map(int, line.split()))
				positive_example.append(parse_line)
		with codecs.open(negative_file, 'r', 'utf-8') as fn:
			for line in fn:
				line = line.strip()
				parse_line = list(map(int, line.split()))
				if len(parse_line) == pm.G_SEQ_LENGTH:
					negative_example.append(parse_line)

		self.sentence = np.array(positive_example + negative_example)

		# Generate labels
		positive_labels = [[0, 1] for _ in positive_example]
		negative_labels = [[1, 0] for _ in negative_example]
		self.labels = np.concatenate([positive_labels, negative_labels], 0)

		# Shuffle the data
		shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
		self.sentence = self.sentence[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		# Split batch
		self.num_batch = int(len(self.labels) / self.batch_size)
		self.sentence = self.sentence[:self.num_batch * self.batch_size]
		self.labels = self.labels[:self.num_batch * self.batch_size]
		self.sentence_batch = np.split(self.sentence, self.num_batch, 0)
		self.labels_batch = np.split(self.labels, self.num_batch, 0)
		self.pointer = 0

	def next_batch(self):
		temp = (self.sentence_batch[self.pointer], self.labels_batch[self.pointer])
		self.pointer = (self.pointer + 1) % self.num_batch
		return temp

	def reset_pointer(self):
		self.pointer = 0


