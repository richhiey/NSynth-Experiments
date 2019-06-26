import tensorflow as tf
import numpy as np
import os

class SequenceEncoder(tf.keras.Model):
	def __init__(self, params):
		self.encoder = build_encoder(params)
		
	def build_encoder(self, params):
		return tf.keras.models.Sequential(
				tf.keras.layers.LSTM()
			)