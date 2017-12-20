# -*- coding:utf-8 -*-
import numpy as np
import codecs
from Config.hyperparameters import Parameters as pm


class Gen_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])
		self.sequence_batch = np.array([])
		self.num_batch = 0
		self.pointer = 0

	def mini_batch(self, data_file):
		token_seqs = []
		with codecs.open(data_file, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				if len(parse_line) == pm.SEQ_LENGTH:
					token_seqs.append(parse_line)

		self.num_batch = int(len(token_seqs) / self.batch_size)
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
		self.token_sentence, self.labels = np.array([]), np.array([])
		self.sentence_batches, self.labels_batches = np.array([]), np.array([])
		self.num_batch = 0
		self.pointer = 0

	def mini_batch(self, positive_file, negative_file):
		positive_example, negative_example = [], []
		with codecs.open(positive_file, 'r', encoding='utf-8') as fpo:
			for line in fpo:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				positive_example.append(parse_line)
		with codecs.open(negative_file, 'r', encoding='utf-8') as fne:
			for line in fne:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				if len(parse_line) == pm.SEQ_LENGTH:
					negative_example.append(parse_line)

		self.token_sentence = np.array(positive_example + negative_example)

		# Generate labels
		positive_labels = [[0, 1] for _ in positive_example]    # one-hot vector = [negative, positive]
		negative_labels = [[1, 0] for _ in negative_example]
		self.labels = np.concatenate([positive_labels, negative_labels], axis=0)

		# Shuffle sampling
		shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
		self.token_sentence = self.token_sentence[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		# Split batches
		self.num_batch = int(len(self.labels) / self.batch_size)
		self.token_sentence = self.token_sentence[:self.num_batch * self.batch_size]
		self.labels = self.labels[:self.num_batch * self.batch_size]
		self.sentence_batches = np.split(self.token_sentence, self.num_batch, 0)
		self.labels_batches = np.split(self.labels, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = (self.sentence_batches[self.pointer], self.labels_batches[self.pointer])
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0


