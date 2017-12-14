# -*- coding:utf-8 -*-
import tensorflow as tf
from Transformer.data_loader import Data_helper
from Transformer.params import Params as pm
from Transformer.modules import Model
from tqdm import tqdm
import codecs, os
from collections import Counter


class Transformer(object):
	def __init__(self, trainable=True):
		self.data_loader = Data_helper()
		self.models = Model()

		if trainable:
			self.x, self.y, self.num_batch = self.data_loader.mini_batch()
		else:
			# Re-initialize
			self.x = tf.placeholder(tf.int32, shape=(None, pm.SEQ_LEN))
			self.y = tf.placeholder(tf.int32, shape=(None, pm.SEQ_LEN))

		# Add 2(<SOS>) in the beginning
		start_token = tf.ones([pm.BATCH_SIZE, 1], dtype=tf.int32) * 2
		self.decoder_inputs = tf.concat((start_token, self.y[:, :-1]), -1)

		# Load vocabulary
		self.de2idx, self.idx2de = self.data_loader.load_vocab(pm.DECODER_VOCAB)
		self.en2idx, self.idx2en = self.data_loader.load_vocab(pm.ENCODER_VOCAB)

# Module -----------------

		# Encoder
		with tf.variable_scope("encoder"):
			# Input Embedding
			self.encoder = self.models.embedding(self.x,
												vocab_size=len(self.en2idx),
												num_units=pm.HIDDEN_UNITS,
												scale=True,
												scope="Input_Embedding")
			print("MSG : Encoder Finished Input embedding!")
			# Positional Encoding
			self.encoder += self.models.positional_encoding(self.x,
															num_units=pm.HIDDEN_UNITS,
															zero_pad=False,
															scale=False,
															scope="en_positional_encoding")
			print("MSG : Encoder Finished Positional embedding!")
			# Dropout
			self.encoder = tf.layers.dropout(self.encoder,
											rate=pm.DROPOUT_RATE,
											training=tf.convert_to_tensor(trainable))
			print("MSG : Encoder Finished dropout!")
			# Body_networks
			for num in range(pm.ENCODER_N):
				with tf.variable_scope("encoder_networds_{}".format(num)):
					# Multi-Head Attention
					self.encoder = self.models.multihead_attention(queries=self.encoder,
																	keys=self.encoder,
																	num_units=pm.HIDDEN_UNITS,
																	num_heads=pm.NUM_HEADS,
																	dropout_rate=pm.DROPOUT_RATE,
																	is_training=trainable,
																	mask=False)

					# Feed Forward
					self.encoder = self.models.feedforward(self.encoder,
															num_units=(4 * pm.HIDDEN_UNITS, pm.HIDDEN_UNITS))
			print("MSG : Encoder Finished Multihead_attention & Feed Forward!")
		# Decoder
		with tf.variable_scope("decoder"):
			# Output Embedding
			self.decoder = self.models.embedding(self.decoder_inputs,
												vocab_size=len(self.en2idx),
												num_units=pm.HIDDEN_UNITS,
												scale=True,
												scope="Output_embedding")
			print("MSG : Decoder Finished Input embedding!")
			# Positional Encoding
			self.decoder += self.models.positional_encoding(self.decoder_inputs,
															num_units=pm.HIDDEN_UNITS,
															zero_pad=False,
															scale=False,
															scope="De_Positional_Encoding")
			print("MSG : Decoder Finished Positional embedding!")
			# Dropout
			self.decoder = tf.layers.dropout(self.decoder,
											rate=pm.DROPOUT_RATE,
											training=tf.convert_to_tensor(trainable))
			print("MSG : Decoder Finished dropout!")
			# Body_networks
			for num in range(pm.DECODER_N):
				with tf.variable_scope("decoder_networks_{}".format(num)):
					# Multi-Head Attention with Mask(self-attention)
					self.decoder = self.models.multihead_attention(queries=self.decoder,
																	keys=self.decoder,
																	num_units=pm.HIDDEN_UNITS,
																	num_heads=pm.NUM_HEADS,
																	dropout_rate=pm.DROPOUT_RATE,
																	is_training=trainable,
																	mask=True,
																	scope="De_Multihead_Attention")

					# Multi-Head Attention(vanilla attention)
					self.decoder = self.models.multihead_attention(queries=self.decoder,
																	keys=self.decoder,
																	num_units=pm.HIDDEN_UNITS,
																	num_heads=pm.NUM_HEADS,
																	dropout_rate=pm.DROPOUT_RATE,
																	is_training=trainable,
																	mask=False,
																	scope="De_Vanilla_Attention")

					# Feed Forward
					self.decoder = self.models.feedforward(self.decoder,
															num_units=[4 * pm.HIDDEN_UNITS, pm.HIDDEN_UNITS])
			print("MSG : Decoder Finished Multihead_attention & Feed Forward!")
		# Linear & Softmax
		self.logits = tf.layers.dense(self.decoder, len(self.en2idx))
		self.predicts = tf.cast(tf.argmax(tf.nn.softmax(self.logits), -1), tf.int32)
		self.is_target = tf.cast(tf.not_equal(self.y, 0), tf.float32)   # Distinguish zero(Padding) or not
		self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predicts, self.y), tf.float32) * self.is_target) / (tf.reduce_sum(self.is_target))
		tf.summary.scalar("accuracy", self.accuracy)

# End Module ----------------

		# Compile
		if trainable:
			# Loss
			self.y_smoothed = self.models.label_smoothing(tf.one_hot(self.y, depth=len(self.en2idx)))
			self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
			self.mean_loss = tf.reduce_mean(self.loss * self.is_target) / (tf.reduce_sum(self.is_target))

			# Optimizer
			self.global_step = tf.Variable(0, name="global_step", trainable=False)  # when it is passed in the minimize() argument list ,the variable is increased by one
			self.optimizer = tf.train.AdamOptimizer(learning_rate=pm.LEARNING_RATE, beta1=0.9, beta2=0.98, epsilon=1e-8).minimize(self.mean_loss, global_step=self.global_step)

			# Summary
			tf.summary.scalar("mean_loss", self.mean_loss)
			self.merged_summary = tf.summary.merge_all()


def build_vocab(path, fname):
	files = codecs.open(path, 'r', encoding='utf-8').read()
	words = files.split()
	wordcount = Counter(words)
	if not os.path.exists("vocabulary"):
		os.mkdir("vocabulary")
	with codecs.open(fname, 'w', encoding='utf-8') as f:
		f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t100000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>"))
		for word, count in wordcount.most_common(len(wordcount)):
			f.write("{}\t{}\n".format(word, count))


def train():
	# Build vocabulary
	build_vocab(pm.SOURCE_TRAIN, pm.ENCODER_VOCAB)
	build_vocab(pm.TARGET_TRAIN, pm.DECODER_VOCAB)
	print("MSG : Finished building vocabulary!")

	# Construct model
	model = Transformer()
	print("MSG : Training transformer ready!")
	init = tf.global_variables_initializer()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Start training
	sv = tf.train.Supervisor(logdir=pm.LOGDIR, init_op=init)
	saver = sv.saver
	with sv.managed_session(config=config) as sess:
		for epoch in range(1, pm.NUM_EPOCH):
			if sv.should_stop():
				break
			for _ in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
				sess.run(model.optimizer)
			print("Loss: {:.6f}, Accuracy: {:.6f}".format(sess.run(model.mean_loss), sess.run(model.accuracy)))

			gs = sess.run(model.global_step)
			saver.save(sess, pm.LOGDIR + "/model_epoch_{}_global_step_{}".format(epoch, gs))

	print("MSG : Done!")


def eval():
	model = Transformer(trainable=False)
	print("MSG : Testing transformer ready!")

	# Load data
	en2idx, idx2en = model.data_loader.load_vocab(pm.ENCODER_VOCAB)
	de2idx, idx2en = model.data_loader.load_vocab(pm.DECODER_VOCAB)

	# Start testing
	sv = tf.train.Supervisor()
	saver = sv.saver
	with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		saver.restore(sess, tf.train.latest_checkpoint(pm.LOGDIR))
		print("MSG : Model restored!")

		# Load Model
		mname = codecs.open(pm.LOGDIR + "/checkpoint", 'r', encoding='utf-8').read().split('"')[1]

		# Inference
		if not os.path.exists("results"):
			os.mkdir("results")
		with codecs.open("results/" + mname, 'w', encoding='utf-8') as f:
			list_of_refs, hypothesis = [], []
			num_batch = model.data_loader.reset_pointer()
			for i in range(num_batch):
				# Get mini batches
				x, sources, targets = model.data_loader.next()

				# Auto-regressive inference

if __name__ == '__main__':
	train()

