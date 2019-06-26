import tensorflow as tf
import numpy as np
import os

class SequenceEncoder(tf.keras.Model):
	def __init__(self, params = {}):
		super(SequenceEncoder, self).__init__(params)
		# Instr embedding size - 16
		self.instr_dim_size = 1006
		self.instr_embedding_size = 16
		# Pitch embedding size - 8
		self.pitch_dim_size = 121
		self.pitch_embedding_size = 8
		# Velocity embedding size - 2
		self.vel_dim_size = 5
		self.vel_embedding_size = 2
		# Time embedding size - 4
		self.time_dim_size = 250
		self.time_embedding_size = 4
		self.original_input_size = self.instr_dim_size + self.pitch_dim_size + self.vel_dim_size + self.time_dim_size 
		self.embedding_input_size = self.instr_embedding_size + self.pitch_embedding_size + self.vel_embedding_size + self.time_embedding_size
		self.sequence_generator = self.build_encoder(params)

	def build_encoder(self, params):
		x = tf.keras.layers.Input(
			shape = (self.time_dim_size, self.embedding_input_size),
			batch_size = 10)
		lstm_1 = tf.keras.layers.LSTM(units = 1024, return_sequences = True)(x)
		lstm_2 = tf.keras.layers.LSTM(units = 1024, return_sequences = True)(lstm_1)
		lstm_3 = tf.keras.layers.LSTM(units = 1024, return_sequences = True)(lstm_2)
		output = tf.keras.layers.Dense(128)(lstm_3)
		model = tf.keras.models.Model(inputs = x, outputs = output)
		return model

	def call(self, input):
		print('Hello world')

x = SequenceEncoder()