# -*- coding:utf-8 -*-
import numpy as np
import codecs
from Config.hyperparameters import Hyperparameter as pm


class Gen_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])
		self.sequence_batch = np.array([])
		self.num_batch = 0
		self.pointer = 0

	def mini_batches(self, data_path):
		token_seqs = []
		with codecs.open(data_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				parse_line = [int(token) for token in line.split()]
				if len(parse_line) == pm.SEQ_LENGTH:
					token_seqs.append(parse_line)

		self.num_batch = len(token_seqs) // self.batch_size
		token_seqs = token_seqs[:self.num_batch * self.batch_size]
		self.token_sentences = np.array(token_seqs)
		self.sequence_batch = np.split(self.token_sentences, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = self.sequence_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch  # back to beginning
		return result

	def reset_pointer(self):
		self.pointer = 0


class Dis_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])
		self.labels = np.array([])
		self.sentence_batches = np.array([])
		self.labels_batches = np.array([])
		self.num_batch = 0
		self.pointer = 0

	def mini_batch(self, positive_path, negative_path):
		positive_samples, negative_samples = [], []
		with codecs.open(positive_path, 'r', encoding='utf-8') as fpo:
			for line in fpo:
				line = line.strip('\n')
				parse_line = [int(token) for token in line.split()]
				positive_samples.append(parse_line)
		with codecs.open(negative_path, 'r', encoding='utf-8') as fne:
			for line in fne:
				line = line.strip('\n')
				parse_line = [int(token) for token in line.split()]
				if len(parse_line) == pm.SEQ_LENGTH:
					negative_samples.append(parse_line)
		self.token_sentences = np.array(positive_samples + negative_samples)

		# Generate labels
		positive_labels = [[0, 1] for _ in range(len(positive_samples))]    # one-hot vector = [negative, positive]
		negative_labels = [[1, 0] for _ in range(len(negative_samples))]
		self.labels = np.concatenate((positive_labels, negative_labels), axis=0)

		# Shuffle sampling
		shuffle_indices = np.arange(len(self.labels))
		np.random.shuffle(shuffle_indices)
		self.token_sentences = self.token_sentences[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		# Split batches
		self.num_batch = len(self.labels) // self.batch_size
		self.token_sentences = self.token_sentences[:self.num_batch * self.batch_size]
		self.labels = self.labels[:self.num_batch * self.batch_size]
		self.sentence_batches = np.split(self.token_sentences, self.num_batch, 0)
		self.labels_batches = np.split(self.labels, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = (self.sentence_batches[self.pointer], self.labels_batches[self.pointer])
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0













