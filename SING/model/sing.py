import tensorflow as tf
import numpy as np
import os
from conv_ae import ConvolutionalEncoder, ConvolutionalDecoder
from sequence_encoder import SequenceEncoder

# Prepare NSynth TFRecordDataset to pass to model
def prepare_nsynth_dataset(dataset):
	def process_parsed_features(point):
	    instrument = tf.one_hot(point['instrument'], 1006)
	    pitch = tf.one_hot(point['pitch'], 21)
	    velocity = tf.one_hot(point['velocity'], 5)
	    audio = point['audio']
	    inputs = tf.concat([instrument, pitch, velocity], axis = 0)
	    return {'inputs': inputs, 'outputs': audio}

	def parse_nsynth(example_proto):
	    features = {
	        "audio": tf.io.FixedLenFeature((4 * 16000), tf.float32),
	        "note": tf.io.FixedLenFeature((), dtype = tf.int64),
	        "note_str": tf.io.FixedLenFeature((), dtype = tf.string),
	        "instrument": tf.io.FixedLenFeature((), dtype = tf.int64),
	        "instrument_str": tf.io.FixedLenFeature((), dtype = tf.string),
	        "instrument_source": tf.io.FixedLenFeature((), dtype = tf.int64),
	        "instrument_source_str": tf.io.FixedLenFeature((), dtype = tf.string),
	        "instrument_family_str": tf.io.FixedLenFeature((), dtype = tf.string),
	        "sample_rate": tf.io.FixedLenFeature((), dtype = tf.int64),
	        "velocity": tf.io.FixedLenFeature((), dtype = tf.int64),
	        "pitch": tf.io.FixedLenFeature((), dtype = tf.int64),
	    }
	    parsed_features = tf.io.parse_single_example(example_proto, features)
	    return process_parsed_features(parsed_features)

	def tfr_dataset_eager(data, batch_size):
	    data = data.apply(tf.data.experimental.shuffle_and_repeat(10000))
	    data = data.apply(tf.data.experimental.map_and_batch(map_func = parse_nsynth, batch_size = batch_size))
	    data = data.prefetch(1)
	    return data

	return tfr_dataset_eager(dataset, 10)


class SINGModel(tf.keras.Model):
	def __init__(self, params = {}):
		super(SINGModel, self).__init__(name = "SINGModel")
		self.conv_encoder = ConvolutionalEncoder()
		self.conv_encoder.build(input_shape=(10,64000,1))
		print(self.conv_encoder.summary())
		self.conv_decoder = ConvolutionalDecoder()
		self.conv_decoder.build(input_shape=(10,250,128))
		print(self.conv_decoder.summary())
		self.sequence_generator = SequenceEncoder()
		self.sequence_generator.build(input_shape=(10,250,30))
		print(self.sequence_generator.summary())

	def call(self):
		print('Hello world')

	def test_network(self):
		test_waveform = np.random.uniform(-1,1,(10,64000,1))
		test_waveform = tf.convert_to_tensor(test_waveform, dtype=tf.float32)
		enc = self.conv_encoder(test_waveform)
		print('Encoded stuff -')
		print(enc)
		dec = self.conv_decoder(enc)
		print('Decoded stuff -')
		print(dec)

	#########################################################
	# - TRAINING STRATEGY
	#
	# - First train the encoder and decoder jointly on a 
	#   reconstruction task with the input waveform
	# - Continue training till convergence
	# - Condition decoder on an SequenceGenerator output (symbols) 
	#   to generate encodings for the decoder
	# - Train till whenever
	# - Done
	#
	########################################################3
	def train(self, dataset_path):
		train_loss_results = []
		train_accuracy_results = []

		nsynth_train_dataset = tf.data.TFRecordDataset(dataset_path)
		nsynth_train_dataset = prepare_nsynth_dataset(nsynth_train_dataset)
		num_epochs = 100
		optimizer = tf.keras.optimizer.Adam()

		def train_step_conv_autoencoder(x, y, loss_type = 'mse'):
			wav = tf.expand_dims(y, axis=-1)
			enc = self.conv_encoder(wav)
			dec = self.conv_decoder(enc)
			if loss == 'spectral_loss':
				print('TODO - implement')
			else:
				loss_fn = tf.keras.losses.MSE()
			with tf.GradientTape() as tape:
				loss = loss_fn(y, dec)
  			return loss, tape.gradient(loss, [self.conv_encoder.trainable_variables,
  				self.conv_decoder.trainable_variables])

		def train_step_sing_model(x, y, loss_type = 'mse'):
			enc = self.sequence_generator(x)
			dec = self.conv_decoder(enc)
			if loss = 'spectral_loss'
				print('TODO - Implement')
			else
				loss_fn = tf.keras.losses.MSE()
			with tf.GradientTape() as tape:
				loss = loss_fn(y, dec)
			return loss, tape.gradient(loss, [self.conv_decoder.trainable_variables,
				self.sequence_generator.trainable_variables])

		# PreTraining Conv Autoencoder
		for epoch in range(num_epochs):
		  # Training loop - using batches of 32
		  for x, y in nsynth_train_dataset:
		    # Optimize the model
		    loss_value, grads = self.train_conv_autoencoder(x, y)
		    optimizer.apply_gradients(zip(grads, [self.conv_encoder.trainable_variables,
				self.conv_decoder.trainable_variables]))
		    print(loss_value)

		# Training SING Model
		for epoch in range(num_epochs):
		  # Training loop - using batches of 32
		  for x, y in nsynth_train_dataset:
		    # Optimize the model
		    loss_value, grads = self.train_sing_model(x, y)
		    optimizer.apply_gradients(zip(grads, [self.conv_decoder.trainable_variables,
				self.sequence_generator.trainable_variables]))
		    print(loss_value)

	def sample(self):
		pass
