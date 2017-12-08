# -*- coding:utf-8 -*-
import tensorflow as tf
from transformer.data_loader import Data_helper
from transformer.params import Params as pm


class Transformer(object):
	def __init__(self, trainable=True):
		self.data_loader = Data_helper()

		if trainable:
			self.x, self.y, self.num_batch = self.data_loader.mini_batch()
		else:
			self.x = tf.placeholder(tf.int32, shape=(None, pm.SEQ_LEN))
			self.y = tf.placeholder(tf.int32, shape=(None, pm.SEQ_LEN))


