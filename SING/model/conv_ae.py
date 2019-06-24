import tensorflow as tf
import numpy as np
import os

class ConvolutionalDecoder(tf.keras.Model):
	def __init__(self, params):
		super(ConvolutionalDecoder, self).__init__(name='ConvolutionalDecoder')
		self.layer_1 = tf.layers.Conv1D()
		self.layer_2 = tf.layers.conv1d()
		self.layer_3 = tf.layers.conv1d()
		self.layer_4 = tf.layers.conv1d()
		self.upsample = tf.layers.

	def call(self, inputs):
		y_1 = self.layer_1(inputs)
		y_2 = self.layer_2(y_1)
		y_3 = self.layer_3(y_2)
		decoded = self.layer_4(y_3)
		return decoded

class ConvolutionalEncoder(tf.keras.Model):
	def __init__(self, params):
		super(ConvolutionalEncoder, self).__init__(name='ConvolutionalEncoder')
		self.layer_1 =  
		self.layer_2 =
		self.layer_3 =
		self.layer_4 =

	def call(self, inputs):
		y_1 = self.layer_1(inputs)
		y_2 = self.layer_2(y_1)
		y_3 = self.layer_3(y_2)
		encoded = self.layer_4(y_3)
		return encoded

class ConvolutionalAutoencoder(tf.keras.Model):
	def __init__(self, params):
		super(ConvolutionalAutoencoder, self).__init__(name='ConvolutionalAutoencoder')
		self.encoder = ConvolutionalEncoder()
		self.decoder = ConvolutionalDecoder()

	def call(self, inputs):
		encoded = self.encoder(inputs)
		decoded = self.decoder(encoded)
		return decoded