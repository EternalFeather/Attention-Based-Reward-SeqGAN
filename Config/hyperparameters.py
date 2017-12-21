# -*- coding:utf-8 -*-


class Parameters(object):

	# Adversarial
	RANDOM_SEED = 88
	START_TOKEN = 0
	BATCH_SIZE = 64
	VOCAB_SIZE = 5000
	GENERATED_NUM = 10000
	K = 3
	TOTAL_BATCHES = 120
	MONTE_CARLO_TURNS = 16
	ADVERSARIAL_DROPOUT = 1.0
	MODEL_PATH = "Log/target_params_py3.pkl"
	REAL_DATA_PATH = "Datasets/Oracle/Real_datasets.txt"
	PRE_GENERATOR_DATA = "Datasets/Oracle/Pre_train_generator_datasets.txt"
	G_NEG_SAMPLING_DATA = "Datasets/Oracle/Generator_negative_sampling_datasets.txt"
	ADVERSARIAL_G_DATA = "Datasets/Oracle/Adversarial_generator_sampling_datasets.txt"
	ADVERSARIAL_NEG_DATA = "Datasets/Oracle/Adversarial_negative_datasets.txt"
	CHINESE_QUATRAINS_FIVE = "Datasets/Chinese_quatrains/Chinese_quatrains_5.txt"
	CHINESE_VOCAB_PATH = "Datasets/Chinese_quatrains/Chinese_quatrains_5.vocab"

	# Generator
	EMB_SIZE = 32
	HIDDEN_SIZE = 32
	SEQ_LENGTH = 20
	LEARNING_RATE = 0.01
	REWARD_GAMMA = 0.95
	G_PRE_TRAIN_EPOCH = 200
	UPDATE_RATE = 0.8
	G_STEP = 1

	# Discriminator
	DIS_EMB_SIZE = 64
	NUM_CLASSES = 2
	FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
	NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
	D_LEARNING_RATE = 1e-4
	L2_REG_LAMBDA = 0.2
	D_PRE_TRAIN_EPOCH = 50
	D_DROP_KEEP_PROB = 0.75
	D_STEP = 5
