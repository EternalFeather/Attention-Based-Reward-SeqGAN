# -*- coding:utf-8 -*-


class Hyperparameter(object):

	# Adversarial
	RANDOM_SEED = 88
	START_TOKEN = 0
	BATCH_SIZE = 64
	VOCAB_SIZE = 5000

	# Generator
	EMB_SIZE = 32
	HIDDEN_SIZE = 32
	SEQ_LENGTH = 20
	LEARNING_RATE = 0.01
	REWARD_GAMMA = 0.95

	# Discriminator
	DIS_EMB_SIZE = 64
	NUM_CLASSES = 2
	FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
	NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
	L2_REG_LAMBDA = 0.2

