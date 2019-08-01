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
        self.sequence_generator.build(input_shape=(10,250,1262))
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
        num_epochs = 50
        optimizer = tf.keras.optimizers.Adam()
        model_checkpoint_dir = 'model_logs_conv_ae/'
        ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = self)
        manager = tf.train.CheckpointManager(ckpt, model_checkpoint_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing model from scratch.")

        def train_step_conv_autoencoder(waveform, optimizer, loss_type = 'MSE'):        
            with tf.GradientTape() as tape:
                encoding = self.conv_encoder(waveform)
                decoding = self.conv_decoder(encoding)
                target_wav = tf.squeeze(waveform, axis=2)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(target_wav, decoding)
                else:
                    print('TODO - spectral_loss')
            trainable_variables = [self.conv_encoder.trainable_variables, self.conv_decoder.trainable_variables]
            flat_list = []
            for sublist in trainable_variables:
                for item in sublist:
                    flat_list.append(item)
            trainable_variables = flat_list
            grads = tape.gradient(loss, trainable_variables)
            grads_and_vars = zip(grads, trainable_variables)
            optimizer.apply_gradients(grads_and_vars)
            return decoding, loss

        def train_step_sing(data, optimizer, loss_type = 'MSE'):        
            inputs = data['inputs']
            waveform = data['outputs']
            with tf.GradientTape() as tape:
                encoding = self.sequence_generator(inputs)
                decoding = self.conv_decoder(encoding)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(waveform, decoding)
                else:
                    print('TODO - spectral_loss')
                trainable_variables = [self.sequence_generator.trainable_variables, self.conv_decoder.trainable_variables]
                flat_list = []
                for sublist in trainable_variables:
                    for item in sublist:
                        flat_list.append(item)
                trainable_variables = flat_list
                grads = tape.gradient(loss, trainable_variables)
                grads_and_vars = zip(grads, trainable_variables)
            optimizer.apply_gradients(grads_and_vars)
            return decoding, loss

#         # PreTraining Conv Autoencoder
        print('Initializing pretraining of Convolutional Autoencoder ...')
        for i in range(num_epochs):
            print('---------- EPOCH ' + str(i) + ' --------------')
            for data in nsynth_train_dataset:
                output_wav, loss = train_step_conv_autoencoder(tf.expand_dims(data['outputs'], axis=-1), optimizer)
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 1000 == 0:
                    print('Generating audio ...')
                    num_samples = 2
                    for j in range(num_samples):
                        print('----- Original waveform -----')
                        IPython.display.display(IPython.display.Audio(data['outputs'][i*i,:], rate=16000))
                        print('----- Ouput waveform -----')
                        IPython.display.display(IPython.display.Audio(output_wav[i*i,:], rate=16000))
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                if int(ckpt.step) % 100 == 0:
                    print(tf.reduce_sum(loss))
            print('---------- EPOCH ' + str(i) + ' END --------------')
        print('Pretraining of Convolutional Autoencoder done!')

        # Training SING Model
        print('Initializing training of SING Model with LSTM sequence generator ...')
        model_checkpoint_dir = 'model_logs_sing/'
        optimizer = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = self)
        manager = tf.train.CheckpointManager(ckpt, model_checkpoint_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        for i in range(num_epochs):
            print('---------- EPOCH ' + str(i) + ' --------------')
            for data in nsynth_train_dataset:
                output_wav, loss = train_step_sing(data, optimizer)
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 1000 == 0:
                    print('Generating audio ...')
                    num_samples = 2
                    for j in range(num_samples):
                        print('----- Inputs -----')
                        print(data['inputs'])
                        print('----- Ouput waveform -----')
                        IPython.display.display(IPython.display.Audio(output_wav[i*i,:], rate=16000))
                        save_path = manager.save()
                        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
 
                if int(ckpt.step) % 100 == 0:
                    print(tf.reduce_sum(loss))
            print('---------- EPOCH ' + str(i) + ' END --------------')
        print('Training of SING Model done!')

    def sample(self):
        pass
