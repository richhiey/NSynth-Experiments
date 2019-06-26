import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import tensorflow as tf
import numpy as np
import itertools
from matplotlib import pyplot as plt

class ConvolutionalEncoder(tf.keras.Model):
    def __init__(self, params):
        super(ConvolutionalEncoder, self).__init__(name = "ConvolutionalEncoder")
        self.model = self.build_encoder(params)

    def build_encoder(self, params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(
            input_shape = params['input_shape'],
            batch_size = params['batch_size']))
        model.add(tf.keras.layers.Conv1D(
            filters = 4096,
            kernel_size = 1024,
            strides = 256,
            padding = 'SAME'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(
            filters = 4096,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(
            filters = 4096,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(
            filters = params['embedding_dim'],
            kernel_size = 1,
            strides = 1,
            padding = 'SAME'))
        return model

    def call(self, x):
        return self.model(x)

class ConvolutionalDecoder(tf.keras.Model):
    def __init__(self, params):
        super(ConvolutionalDecoder, self).__init__(name = "ConvolutionalDecoder")
        self.model = self.build_decoder(params)

    def build_decoder(self, params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(
            input_shape = params['input_shape']))
        model.add(tf.keras.layers.Conv1D(
            filters = 4096,
            kernel_size = 9,
            strides = 1,
            padding = 'SAME'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(
            filters = 4096,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(
            filters = 4096,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis = 1)
        ))
        model.add(tf.keras.layers.Conv2DTranspose(
            filters = 1,
            kernel_size = (1,1024),
            strides = (1,256),
            padding = "SAME"))
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis = [1,3])
        ))
        return model

    def call(self, x):
        return self.model(x)


class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, params):
        super(ConvolutionalAutoencoder, self).__init__(params)
        self.max_steps = params['max_steps']
        self.learning_rate = params['learning_rate']
        self.epochs = params['epochs']
        print('Building Encoder ..')
        self.inference_net = ConvolutionalEncoder(params['encoder'])
        print('Building Decoder ..')
        self.generative_net = ConvolutionalDecoder(params['decoder'])

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z, apply_sigmoid = False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


    def train_step(self, waveform, loss_type):
        optimizer = tf.keras.optimizer.Adam()
        
        if loss_type == 'MSE':
            compute_loss = tf.keras.losses.MSE()
        else:
            # Op to calculate spectral loss between two waveformss
            print('TODO - spectral_loss')
        encoding = self.inference_net(waveform)
        decoding = self.generative_net(encoding)

        loss = compute_loss(waveform, decoding)
        optimizer.minimize(loss)
        return loss

    def train(self, dataset_path):
        dataset = tf.data.TFRecordDataset(dataset_path)
        for i in range(self.epochs):
            print('---------- EPOCH ' + str(i) + ' --------------')
            for waveform in dataset:
                print(waveform)
                self.train_step(waveform)
            print('---------- EPOCH ' + str(i) + ' END --------------')


    def call(self, params):
      print('Hello world')

params = {
    'max_steps': 100,
    'epochs': 10,
    'learning_rate': 0.0001,
    'encoder': {
        'input_shape': (64000, 1),
        'batch_size': 10,
        'embedding_dim': 128
    },
    'decoder': {
        'input_shape': (250, 128)
    }
}

cae = ConvolutionalAutoencoder(params)
print(cae.summary())
print(cae.inference_net.summary())
print(cae.generative_net.summary())

dataset_path = ''
cae.train(dataset_path)