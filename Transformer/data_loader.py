# -*- coding:utf-8 -*-
import codecs
import numpy as np
import sys
import tensorflow as tf
from Transformer.params import Params as pm


class Data_helper(object):
	def __init__(self):
		self.train_num_batch = 0
		self.test_num_batch = 0
		self.pointer = 0
		self.vocab = []
		self.en_sents, self.de_sents = [], []
		self.xtoken_list, self.ytoken_list, self.Source, self.Targets = [], [], [], []
		self.word2idx, self.idx2word = {}, {}

	def mini_batch(self):
		X, Y, _, _ = self.load_datasets("train")
		self.train_num_batch = len(X) // pm.BATCH_SIZE
		X = tf.convert_to_tensor(X, tf.int32)
		Y = tf.convert_to_tensor(Y, tf.int32)

		# Input Queue by CPU
		input_queues = tf.train.slice_input_producer([X, Y])
		# Get mini batch from Queue
		x, y = tf.train.shuffle_batch(input_queues,
									num_threads=8,
									batch_size=pm.BATCH_SIZE,
									capacity=pm.BATCH_SIZE * 64,    # Max_number of batches in queue
									min_after_dequeue=pm.BATCH_SIZE * 32,   # Min_number of batches in queue after dequeue
									allow_smaller_final_batch=False)

		return x, y, self.train_num_batch

	def load_datasets(self, type_name):
		if type_name == "train":
			self.de_sents = [line for line in codecs.open(pm.SOURCE_TRAIN, 'r', encoding='utf-8').read().split('\n') if line]
			self.en_sents = [line for line in codecs.open(pm.TARGET_TRAIN, 'r', encoding='utf-8').read().split('\n') if line]
		elif type_name == "test":
			self.de_sents = [line for line in codecs.open(pm.SOURCE_TEST, 'r', encoding='utf-8').read().split('\n') if line]
			self.en_sents = [line for line in codecs.open(pm.TARGET_TEST, 'r', encoding='utf-8').read().split('\n') if line]
		else:
			print("MSG : Error from load_datasets.")
			sys.exit(0)
		x, y, sources, targets = self.generate(self.de_sents, self.en_sents)

		return x, y, sources, targets

	def generate(self, de_sent, en_sent):
		de2idx, idx2de = self.load_vocab(pm.DECODER_VOCAB)
		en2idx, idx2en = self.load_vocab(pm.ENCODER_VOCAB)

		for source_sent, target_sent in zip(en_sent, de_sent):
			x = [en2idx.get(word, 1) for word in (source_sent + " <EOS>").split()]
			y = [de2idx.get(word, 1) for word in (target_sent + " <EOS>").split()]
			if max(len(x), len(y)) <= pm.SEQ_LEN:
				self.xtoken_list.append(np.array(x))
				self.ytoken_list.append(np.array(y))
				self.Source.append(source_sent)
				self.Targets.append(target_sent)

		# Padding 0(<PAD>)
		x_np = np.zeros([len(self.xtoken_list), pm.SEQ_LEN], dtype=np.int32)
		y_np = np.zeros([len(self.ytoken_list), pm.SEQ_LEN], dtype=np.int32)
		for i, (x, y) in enumerate(zip(self.xtoken_list, self.ytoken_list)):
			x_np[i] = np.lib.pad(x, [0, pm.SEQ_LEN - len(x)], 'constant', constant_values=(0, 0))
			y_np[i] = np.lib.pad(y, [0, pm.SEQ_LEN - len(y)], 'constant', constant_values=(0, 0))

		return x_np, y_np, self.Source, self.Targets

	def load_vocab(self, file):
		self.vocab = [line.split()[0] for line in codecs.open(file, 'r', encoding='utf-8').read().splitlines() if int(line.split()[1]) >= pm.MIN_WORD_COUNT]
		self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
		self.idx2word = {self.word2idx[word]: word for word in self.word2idx}

		return self.word2idx, self.idx2word

	def next(self):
		X, _, Sources, Targets = self.load_datasets("test")
		self.test_num_batch = len(X) // pm.BATCH_SIZE
		x = X[self.pointer * pm.BATCH_SIZE: (self.pointer + 1) * pm.BATCH_SIZE]
		sources = Sources[self.pointer * pm.BATCH_SIZE: (self.pointer + 1) * pm.BATCH_SIZE]
		targets = Targets[self.pointer * pm.BATCH_SIZE: (self.pointer + 1) * pm.BATCH_SIZE]
		self.pointer = (self.pointer + 1) % self.test_num_batch
		return x, sources, targets

	def reset_pointer(self):
		self.pointer = 0
		return self.test_num_batch
