import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import itertools
from matplotlib import pyplot as plt
import IPython

class ConvolutionalEncoder(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__(name = "ConvolutionalEncoder")
        self.model = self.build_encoder()
        print(self.model.summary())

    def build_encoder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(
            input_shape = (64000, 1)))
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
            filters = 128,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME'))
        return model

    def call(self, x):
        return self.model(x)

class ConvolutionalDecoder(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalDecoder, self).__init__(name = "ConvolutionalDecoder")
        self.model = self.build_decoder()
        print(self.model.summary())

    def build_decoder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(
            input_shape = (250, 128)))
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
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.learning_rate = 0.0003
        self.epochs = 50
        print('Building Encoder ..')
        self.inference_net = ConvolutionalEncoder()
        print('Building Decoder ..')
        self.generative_net = ConvolutionalDecoder()

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z, apply_sigmoid = False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


    def train_step(self, waveform, loss_type = 'MSE'):        
        with tf.GradientTape() as tape:
            encoding = self.inference_net(waveform)
            decoding = self.generative_net(encoding)
            target_wav = tf.squeeze(waveform, axis=2)
            if loss_type == 'MSE':
                loss = tf.keras.losses.MSE(target_wav, decoding)
            else:
                print('TODO - spectral_loss')
        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = zip(grads, self.trainable_variables)
        return decoding, loss, grads_and_vars

    def train(self, dataset):
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        model_checkpoint_dir = 'model_logs/'
        ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = self)
        manager = tf.train.CheckpointManager(ckpt, model_checkpoint_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing model from scratch.")
        for i in range(self.epochs):
            print('---------- EPOCH ' + str(i) + ' --------------')
            for data in dataset:
                output_wav, loss, grads = self.train_step(tf.expand_dims(data['outputs'], axis=-1))
                optimizer.apply_gradients(grads)
                tf.summary.scalar('Conv_AE_Loss', tf.reduce_sum(loss), step=int(ckpt.step))
                print(tf.reduce_sum(loss))
                if int(ckpt.step) % 1000 == 0:
                    print('Generating audio ...')
                    num_samples = 10
                    for j in range(num_samples):
                        IPython.display.display(IPython.display.Audio(output_wav[i,:], rate=16000))
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                ckpt.step.assign_add(1)
            print('---------- EPOCH ' + str(i) + ' END --------------')


    def call(self):
      print('Hello world')