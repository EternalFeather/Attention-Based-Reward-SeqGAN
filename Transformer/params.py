# -*- coding:utf-8 -*-


class Params(object):

	SOURCE_TRAIN = "corpora/train_src.txt"
	TARGET_TRAIN = "corpora/train_tgt.txt"
	SOURCE_TEST = "corpora/test_src.txt"
	TARGET_TEST = "corpora/test_tgt.txt"
	DECODER_VOCAB = "vocabulary/de.vocab.tsv"
	ENCODER_VOCAB = "vocabulary/en.vocab.tsv"

	BATCH_SIZE = 32
	SEQ_LEN = 10
	MIN_WORD_COUNT = 20
	ENCODER_N = 6
	DECODER_N = 6
	LEARNING_RATE = 0.0001
	HIDDEN_UNITS = 512
	DROPOUT_RATE = 0.1
	NUM_HEADS = 8
	LOGDIR = "logdir"
	NUM_EPOCH = 20

