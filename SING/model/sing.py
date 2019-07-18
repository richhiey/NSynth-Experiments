import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
from .conv_ae import ConvolutionalEncoder, ConvolutionalDecoder
from .sequence_encoder import SequenceEncoder
import IPython

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
    def train(self, nsynth_train_dataset):
        num_epochs = 100
        optimizer = tf.keras.optimizers.Adam()
#         model_checkpoint_dir = 'model_logs_conv_ae/'
#         ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = self.conv_encoder])
#         manager = tf.train.CheckpointManager(ckpt, model_checkpoint_dir, max_to_keep=3)
#         ckpt.restore(manager.latest_checkpoint)
#         if manager.latest_checkpoint:
#             print("Restored from {}".format(manager.latest_checkpoint))
#         else:
#             print("Initializing model from scratch.")

        def train_step_conv_autoencoder(waveform, optimizer, loss_type = 'MSE'):        
            with tf.GradientTape() as tape:
                encoding = self.conv_encoder(waveform)
                decoding = self.conv_decoder(encoding)
                target_wav = tf.squeeze(waveform, axis=2)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(target_wav, decoding)
                else:
                    print('TODO - spectral_loss')
                grads = tape.gradient(loss, [self.conv_encoder.trainable_variables])
                grads_and_vars = zip(grads, [self.conv_encoder.trainable_variables])
            optimizer.apply_gradients(grads_and_vars)
            return decoding, loss, grads_and_vars

        def train_step_sing(waveform, optimizer, loss_type = 'MSE'):        
            with tf.GradientTape() as tape:
                encoding = self.inference_net(waveform)
                decoding = self.generative_net(encoding)
                target_wav = tf.squeeze(waveform, axis=2)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(target_wav, decoding)
                else:
                    print('TODO - spectral_loss')
                grads = tape.gradient(loss, [self.inference_net.trainable_variables])
                grads_and_vars = zip(grads, [self.inference_net.trainable_variables])
            print(grads_and_vars)
            optimizer.apply_gradients(grads_and_vars)
            return decoding, loss

        # PreTraining Conv Autoencoder
        for i in range(num_epochs):
            print('---------- EPOCH ' + str(i) + ' --------------')
            for data in nsynth_train_dataset:
                output_wav, loss = train_step_conv_autoencoder(tf.expand_dims(data['outputs'], axis=-1), optimizer)
                if int(ckpt.step) % 100 == 0:
                    print('Generating audio ...')
                    print(loss)
                    IPython.display.display(IPython.display.Audio(output_wav, rate=16000))
                    
#                 if int(ckpt.step) % 1000 == 0:
#                     save_path = manager.save()
#                     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print('---------- EPOCH ' + str(i) + ' END --------------')

        # Training SING Model
#         model_checkpoint_dir = 'model_logs_sing/'
#         optimizer = tf.keras.optimizers.Adam()
#         ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = [self.conv_encoder, self.conv_decoder])
#         manager = tf.train.CheckpointManager(ckpt, model_checkpoint_dir, max_to_keep=3)
#         ckpt.restore(manager.latest_checkpoint)
        for i in range(num_epochs):
            print('---------- EPOCH ' + str(i) + ' --------------')
            for data in nsynth_train_dataset:
                output_wav, loss = train_step_sing(tf.expand_dims(data['outputs'], axis=-1), optimizer)
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 100 == 0:
                    print('Generating audio ...')
                    print(loss)
                    IPython.display.display(IPython.display.Audio(output_wav, rate=16000))
                    
#                 if int(ckpt.step) % 1000 == 0:
#                     save_path = manager.save()
#                     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print('---------- EPOCH ' + str(i) + ' END --------------')

    def sample(self):
        pass
