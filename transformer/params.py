# -*- coding:utf-8 -*-


class Params(object):

	SOURCE_TRAIN = "corpora/train_src.txt"
	TARGET_TRAIN = "corpora/train_tgt.txt"
	SOURCE_TEST = "corpora/test_src.txt"
	TARGET_TEST = "corpora/test_tgt.txt"
	DECODER_VOCAB = "Vocabulary/de.vocab.tsv"
	ENCODER_VOCAB = "Vocabulary/en.vocab.tsv"

	BATCH_SIZE = 32
	SEQ_LEN = 10
	MIN_WORD_COUNT = 20