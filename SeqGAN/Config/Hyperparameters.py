# -*- coding:utf-8 -*-


class Parameters():

	# General
	SEED = 88
	START_TOKEN = 0
	BATCH_SIZE = 64
	VOCAB_SIZE = 5000
	L2_REG_LAMBDA = 0.0
	GENERATED_NUM = 10000
	POSITIVE_FILE = 'saver/real_data.txt'
	EVAL_FILE = 'saver/eval_file.txt'
	NEG_FILE = 'saver/generator_sample.txt'
	TOTAL_BATCH = 200

	# Generator Hyper-parameters
	G_EMB_DIM = 32 			# Embedding dimension
	G_HIDDEN_DIM = 32 		# Hidden state dimension of lstm cell
	G_SEQ_LENGTH = 20			# Sequence length
	G_PRE_EPOCH_NUM = 200 	# Supervise epochs with maximum likelihood estimation
	G_LEARNING_RATE = 0.01 	# Learning rate
	G_REWARD_GAMMA = 0.95		# Gamma parameter of reinforcement learning

	# Discriminator Hyper-parameters
	D_SEQ_LENGTH = 20
	D_NUM_CLASSES = 2
	D_EMB_DIM = 64
	D_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
	D_HIDDEN_SIZES = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] 
	D_DROPOUT_KEEP_PROB = 0.75
	D_LEARNING_RATE = 1e-4
	D_L2_REG_LAMBDA = 0.2
	D_PRE_EPOCH_NUM = 50
	D_NLL_K = 3
	D_UPDATE_STEP = 5
