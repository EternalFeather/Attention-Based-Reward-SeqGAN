# -*- coding:utf-8 -*-
import random
from SeqGAN.Config.hyperparameters import Hyperparameter as pm
import numpy as np
import tensorflow as tf
from SeqGAN.Datasets.data_loader import Gen_data_loader, Dis_data_loader
from SeqGAN.Model.generator import Generator
from SeqGAN.Model.discriminator import Discriminator
import pickle
from SeqGAN.Model.corpus_lstm import Corpus_lstm
import codecs


class SeqGAN(object):
	def __init__(self):
		random.seed(pm.RANDOM_SEED)
		np.random.seed(pm.RANDOM_SEED)
		assert pm.START_TOKEN == 0

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())

		# Init
		gen_data_loader = Gen_data_loader(pm.BATCH_SIZE)
		likelihood_data_loader = Gen_data_loader(pm.BATCH_SIZE)     # For Testing
		dis_data_loader = Dis_data_loader(pm.BATCH_SIZE)
		generator = Generator(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.EMB_SIZE, pm.HIDDEN_SIZE, pm.SEQ_LENGTH, pm.START_TOKEN,pm.LEARNING_RATE, pm.REWARD_GAMMA)
		discriminator = Discriminator(pm.SEQ_LENGTH, pm.NUM_CLASSES, pm.VOCAB_SIZE, pm.DIS_EMB_SIZE, pm.FILTER_SIZES, pm.NUM_FILTERS, pm.L2_REG_LAMBDA)
		target_params = pickle.load(open('SeqGAN/Model/target_params.pkl'))     # Oracle LSTM_model for corpus generation
		corpus_lstm = Corpus_lstm(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.EMB_SIZE, pm.HIDDEN_SIZE, pm.SEQ_LENGTH, pm.START_TOKEN, target_params)

		# Generate 1W sequences of length 20 as the training set S for the generator model
		self.generate_samples(sess, corpus_lstm, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.REAL_DATA_PATH)
		gen_data_loader.create_batches(pm.REAL_DATA_PATH)

	def generate_samples(self, sess, trainable_model, batch_size, generated_num, output_path):
		generated_samples = []
		total_num = generated_num // batch_size
		for _ in range(total_num):
			generated_samples.extend(trainable_model.generate(sess))

		with codecs.open(output_path, 'w', encoding='utf-8') as fout:
			for data in generated_samples:
				buffer = " ".join(str(x) for x in data)  # write in token format
				fout.write(buffer + '\n')

	def target_loss(self):
		pass

	def pre_train_epoch(self):
		pass

if __name__ == '__main__':
	model = SeqGAN()
