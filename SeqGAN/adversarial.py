# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import random
import os
from SeqGAN.Config.Hyperparameters import Parameters as pm
from SeqGAN.Datasets.Dataloader import Gen_dataloader, Dis_dataloader
from SeqGAN.Model.Generator import Generator
from SeqGAN.Model.Discriminator import Discriminator
# from LSTM import Target_lstm
from SeqGAN.Model.target_lstm import TARGET_LSTM as Target_lstm
from SeqGAN.Model.Rollout import Rollout
import pickle
import codecs
from matplotlib import pyplot as plt


def generator_samples(sess, trainable_model, batch_size, generated_num, output_file):
	'''
	The Generator samples from oracle model.
	'''
	generator_samples = []
	for _ in range(int(generated_num / batch_size)):
		generator_samples.extend(trainable_model.generate(sess))

	with codecs.open(output_file, 'w', 'utf-8') as f:
		for poem in generator_samples:
			buffer = ' '.join([str(i) for i in poem])
			f.write(buffer + '\n')


def target_loss(sess, target_lstm, data_loader):
	'''
	The oracle negative log-likelihood tested with the oracle model "target_lstm"
	'''
	nll = []
	data_loader.reset_pointer()

	for i in range(data_loader.num_batch):
		batch = data_loader.next_batch()
		g_loss = sess.run(target_lstm.pretrain_loss, feed_dict={target_lstm.x: batch})
	nll.append(g_loss)

	return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
	'''
	Pre-train the generator using MLE for one-epoch
	'''
	supervised_g_losses = []
	data_loader.reset_pointer()

	for i in range(data_loader.num_batch):
		batch = data_loader.next_batch()
		_, g_loss = trainable_model.pretrain_step(sess, batch)
		supervised_g_losses.append(g_loss)

	return np.mean(supervised_g_losses)


if __name__ == '__main__':
	random.seed(pm.SEED)
	np.random.seed(pm.SEED)
	assert pm.START_TOKEN == 0

	gen_data_loader = Gen_dataloader(pm.BATCH_SIZE)
	likelihood_data_loader = Gen_dataloader(pm.BATCH_SIZE)  # Standard way
	dis_data_loader = Dis_dataloader(pm.BATCH_SIZE)

	# Init generator(including Pre-train)
	generator = Generator(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.G_EMB_DIM, pm.G_HIDDEN_DIM, \
	                      pm.G_SEQ_LENGTH, pm.START_TOKEN, pm.G_LEARNING_RATE, pm.G_REWARD_GAMMA)

	discriminator = Discriminator(pm.D_SEQ_LENGTH, pm.D_NUM_CLASSES, pm.D_NUM_CLASSES, pm.D_EMB_DIM, \
	                              pm.D_FILTER_SIZES, pm.D_HIDDEN_SIZES, pm.D_LEARNING_RATE, pm.D_L2_REG_LAMBDA)

	target_params = pickle.load(codecs.open('saver/target_params.pkl'))
	target_lstm = Target_lstm(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.G_EMB_DIM, pm.G_HIDDEN_DIM, pm.G_SEQ_LENGTH, \
	                          pm.START_TOKEN, target_params)

	# Init graph
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())

	# Use oracle model to provide positive examples
	if not os.path.exists(pm.POSITIVE_FILE):
		os.mkdir(pm.POSITIVE_FILE)
	generator_samples(sess, target_lstm, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.POSITIVE_FILE)
	gen_data_loader.create_batch(pm.POSITIVE_FILE)

	log = codecs.open('saver/experiment-log.txt', 'w', 'utf-8')
	temp, idx = [], []
	# Pre-training generator
	print('MSG : Start pre-training generator...')
	log.write('pre-training...\n')

	for epoch in range(pm.G_PRE_EPOCH_NUM):
		loss = pre_train_epoch(sess, generator, gen_data_loader)
		if epoch % 5 == 0:
			generator_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.EVAL_FILE)
			likelihood_data_loader.create_batch(pm.EVAL_FILE)
			test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
			temp.append(test_loss)
			idx.append(epoch)
			print('pre-train epoch: {}, test_loss: {}'.format(epoch + 1, test_loss))
			buffer = 'epoch:\t' + str(epoch + 1) + '\tnll:\t' + str(test_loss) + '\n'
			log.write(buffer)

	# plt.figure()
	# plt.plot(idx, temp, color = "red")
	# plt.show()

	# pre-training discriminator
	print('MSG : Start pre-training discriminator...')
	for _ in range(pm.D_PRE_EPOCH_NUM):
		generator_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.NEG_FILE)
		dis_data_loader.create_batch(pm.POSITIVE_FILE, pm.NEG_FILE)
		for _ in range(pm.D_NLL_K):
			dis_data_loader.reset_pointer()
			for i in range(dis_data_loader.num_batch):
				x_batch, y_batch = dis_data_loader.next_batch()
				sess.run(discriminator.train_op, feed_dict={discriminator.input_x: x_batch,
															discriminator.input_y: y_batch,
															discriminator.dropout_keep_prob: pm.D_DROPOUT_KEEP_PROB})

	rollout = Rollout(generator, 0.8)

	print('MSG : Start Adversarial Training...')
	log.write('adversarial training...\n')
	for total_batch in range(pm.TOTAL_BATCH):
		# Train the generator for one step
		for i in range(1):
			samples = generator.generate(sess)
			rewards = rollout.get_reward(sess, samples, 16, discriminator)
			sess.run(generator.g_updates, feed_dict={generator.x: samples, generator.rewards: rewards})

		# Test
		if total_batch % 5 == 0 or total_batch == pm.TOTAL_BATCH - 1:
			generator_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.EVAL_FILE)
			likelihood_data_loader.create_batch(pm.EVAL_FILE)
			test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
			temp.append(test_loss)
			idx.append(pm.G_PRE_EPOCH_NUM + total_batch)
			buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
			print('total_batch: {}, test_loss: {}'.format(total_batch, test_loss))
			log.write(buffer)
		# Update the discriminator
		rollout.update_params()

		# Train the discriminator
		for _ in range(pm.D_UPDATE_STEP):
			generator_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.NEG_FILE)
			dis_data_loader.create_batch(pm.POSITIVE_FILE, pm.NEG_FILE)

			for _ in range(pm.D_NLL_K):
				dis_data_loader.reset_pointer()
				for i in range(dis_data_loader.num_batch):
					x_batch, y_batch = dis_data_loader.next_batch()
					sess.run(discriminator.train_op, feed_dict={discriminator.input_x: x_batch,
																discriminator.input_y: y_batch,
																discriminator.dropout_keep_prob: pm.D_DROPOUT_KEEP_PROB})
	log.close()

	plt.figure()
	plt.plot(idx, temp, color="red")
	plt.show()
