# -*- coding:utf-8 -*-
import numpy as np


class Gen_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = []


class Dis_data_loader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])
		self.labels = np.array([])
