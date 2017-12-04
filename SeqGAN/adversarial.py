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
from SeqGAN.Model.reinforcement import Reinforcement
import codecs
import matplotlib.pyplot as plt


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
		target_params = pickle.load(open("SeqGAN/Model/target_params.pkl"))     # Oracle LSTM_model for corpus generation
		corpus_lstm = Corpus_lstm(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.EMB_SIZE, pm.HIDDEN_SIZE, pm.SEQ_LENGTH, pm.START_TOKEN, target_params)

		# Generate 1W sequences of length 20 as the training set S for the generator model
		self.generate_samples(sess, corpus_lstm, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.REAL_DATA_PATH)
		gen_data_loader.mini_batches(pm.REAL_DATA_PATH)

		log = codecs.open("Log/experiment-log.txt", 'w', encoding='utf-8')

		# Pre-train Generator
		temp = []
		print("MSG : Start Pre-train Generator...")
		log.write("Pre-train Generator...\n")
		for epoch in range(pm.G_PRE_TRAIN_EPOCH):
			pretrain_loss = self.gen_pre_train_loss(sess, generator, gen_data_loader)
			if epoch % 5 == 0:
				self.generate_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.PRE_GENERATOR_DATA)
				likelihood_data_loader.mini_batches(pm.PRE_GENERATOR_DATA)
				test_loss = self.target_loss(sess, corpus_lstm, likelihood_data_loader)
				temp.append(test_loss)
				print("Pre-train Gen epoch: {}, Test_loss: {}, Pretrain_loss: {}".format(epoch + 1, test_loss, pretrain_loss))
				buffer = "Pre-train Generator Epoch:\t{}\tNLL:\t{}\tGenerator_Loss:{}\n".format(str(epoch + 1), str(test_loss), str(pretrain_loss))
				log.write(buffer)

		plt.figure()
		plt.plot(temp)
		plt.ion()
		plt.show()

		# Pre-train Discriminator
		temp = []
		print("MSG : Start Pre-train Discriminator...")
		log.write("Pre-train Discriminator...\n")
		for epoch in range(pm.D_PRE_TRAIN_EPOCH):
			self.generate_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.G_NEG_SAMPLING_DATA)
			dis_data_loader.mini_batch(pm.REAL_DATA_PATH, pm.G_NEG_SAMPLING_DATA)
			test_loss = 0.0
			for _ in range(pm.K):
				test_loss = self.dis_pre_train_loss(sess, discriminator, dis_data_loader)

			if epoch % 5 == 0:
				temp.append(test_loss)
				print("Pre-train Dis epoch: {}, Test_loss: {}".format(epoch + 1, test_loss))
				buffer = "Pre-train Discriminator Epoch:\t{}\tNLL:\t{}\n".format(str(epoch + 1), str(test_loss))
				log.write(buffer)

		plt.figure()
		plt.plot(temp)
		plt.ion()
		plt.show()

		reinforcement = Reinforcement(generator, pm.UPDATE_RATE)

		# Adversarial Train
		print("MSG : Start Adversarial Training...")
		log.write("Adversarial Training...\n")
		for total_batch in range(pm.TOTAL_BATCHES):
			# Train the generator for one step
			for i in range(pm.G_STEP):
				samples = generator.generate(sess)
				rewards = reinforcement.get_reward(sess, samples, pm.GIVEN_NUM, discriminator)






	def generate_samples(self, sess, trainable_model, batch_size, generated_num, output_path):
		generated_samples = []
		total_num = generated_num // batch_size
		for _ in range(total_num):
			generated_samples.extend(trainable_model.generate(sess))

		with codecs.open(output_path, 'w', encoding='utf-8') as fout:
			for data in generated_samples:
				buffer = " ".join(str(x) for x in data)  # write in token format
				fout.write(buffer + '\n')

	def target_loss(self, sess, target_model, data_loader):
		nll = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			batch = data_loader.next_batch()
			loss = sess.run(target_model.loss, feed_dict={target_model.x: batch})
			nll.append(loss)

		return np.mean(nll)

	def gen_pre_train_loss(self, sess, trainable_model, data_loader):
		supervised_g_losses = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			batch = data_loader.next_batch()
			_, g_loss = trainable_model.pretrain_forward(sess, batch)
			supervised_g_losses.append(g_loss)

		return np.mean(supervised_g_losses)

	def dis_pre_train_loss(self, sess, trainable_model, data_loader):
		supervised_d_losses = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			x_batch, y_batch = data_loader.next_batch()
			_, d_loss = sess.run(trainable_model.pretrain_forward(sess, x_batch, y_batch, pm.D_DROP_KEEP_PROB))
			supervised_d_losses.append(d_loss)

		return np.mean(supervised_d_losses)

if __name__ == '__main__':
	model = SeqGAN()
