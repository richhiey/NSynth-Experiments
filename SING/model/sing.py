import os
import tensorflow as tf
import numpy as np
from .conv_ae import ConvolutionalEncoder, ConvolutionalDecoder
from .sequence_encoder import SequenceEncoder
import utils

class SINGModel(tf.keras.Model):
    def __init__(self, params = {}):
        super(SINGModel, self).__init__(name = "SINGModel")
        # Model variables
        self.conv_encoder = ConvolutionalEncoder()
        self.conv_encoder.build(input_shape=(10,64000,1))
        print(self.conv_encoder.summary())
        self.conv_decoder = ConvolutionalDecoder()
        self.conv_decoder.build(input_shape=(10,250,128))
        print(self.conv_decoder.summary())
        self.sequence_generator = SequenceEncoder()
        self.sequence_generator.build(input_shape=(10,250,1262))
        print(self.sequence_generator.summary())

        # Extra variables
        self.learning_rate = 0.0003
        self.epochs = 50
        self.conv_ae_model_log_dir = 'model_logs_sing_conv_ae/'
        self.sing_model_log_dir = 'model_logs_sing'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.num_steps_checkpoint = 1000
        self.num_outputs = 5
        self.sampling_rate = 16000

    def call(self, inputs):
        return self.conv_decoder(self.sequence_generator(inputs))

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
    def train(self, dataset):

        def train_step_conv_autoencoder(waveform, loss_type = 'MSE'):        
            with tf.GradientTape() as tape:
                encoding = self.conv_encoder(waveform)
                decoding = self.conv_decoder(encoding)
                target_wav = tf.squeeze(waveform, axis = 2)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(target_wav, decoding)
                else:
                    loss = utils.losses.spectral_loss(data['outputs'], decoding)
            trainable_sub_models = [self.conv_encoder, self.conv_decoder]
            trainable_variables = utils.processing.get_all_trainable_variables(
                trainable_sub_models
            )
            grads = tape.gradient(loss, trainable_variables)
            grads_and_vars = zip(grads, trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)
            return decoding, loss

        def train_step_sing(data, loss_type = 'MSE'):
            with tf.GradientTape() as tape:
                encoding = self.sequence_generator(data['inputs'])
                decoding = self.conv_decoder(encoding)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(data['outputs'], decoding)
                else:
                    loss = utils.losses.spectral_loss(data['outputs'], decoding)
                trainable_sub_models = [self.sequence_generator, self.conv_decoder]
                trainable_variables = utils.processing.get_all_trainable_variables(
                    trainable_sub_models
                )
                grads = tape.gradient(loss, trainable_variables)
                grads_and_vars = zip(grads, trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)
            return decoding, loss

        # PreTraining Conv Autoencoder
        ckpt, manager = utils.processing.get_tensorflow_checkpoint(
            self.optimizer,
            self.conv_ae_model_log_dir
        )
        print('Initializing pretraining of Convolutional Autoencoder ...')
        for i in range(self.num_epochs):
            print('-------------------- EPOCH ' + str(i) + ' ------------------------')
            for data in dataset:
                output_wav, loss = train_step_conv_autoencoder(
                    tf.expand_dims(data['outputs'], axis = -1)
                )
                step = int(ckpt.step)
                utils.tensorboard.log_stuff_to_tensorboard(
                    step,
                    tf.reduce_sum(loss),
                    grads
                )
                if step % self.num_steps_checkpoint == 0:
                    print("============== STEP " + str(step) + " ==============")
                    utils.processing.log_statistics_to_console(
                        tf.reduce_sum(loss)
                    )
                    utils.processing.log_outputs_to_notebook(
                        data['outputs'],
                        output_wav,
                        num_outputs = self.num_outputs,
                        audio_sampling_rate = self.sampling_rate
                    )
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(step, save_path))
                    print("============== STEP END ==============")
                ckpt.step.assign_add(1)
            print('-------------------- EPOCH ' + str(i) + ' END ------------------------')
        print('Pretraining of Convolutional Autoencoder done!')

        # Training SING Model
        print('Initializing training of SING Model with LSTM sequence generator ...')
        ckpt, manager = utils.processing.get_tensorflow_checkpoint(
            self.optimizer,
            self.sing_model_log_dir
        )
        for i in range(self.num_epochs):
            print('-------------------- EPOCH ' + str(i) + ' ------------------------')
            for data in dataset:
                output_wav, loss = train_step_sing(data)
                step = int(ckpt.step)
                utils.tensorboard.log_stuff_to_tensorboard(
                    step,
                    tf.reduce_sum(loss),
                    grads
                )
                if step % self.num_steps_checkpoint == 0:
                    print("============== STEP " + str(step) + " ==============")
                    utils.processing.log_statistics_to_console(
                        tf.reduce_sum(loss)
                    )
                    utils.processing.log_outputs_to_notebook(
                        data['outputs'],
                        output_wav,
                        num_outputs = self.num_outputs,
                        audio_sampling_rate = self.sampling_rate
                    )
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(step, save_path))
                    print("============== STEP END ==============")
                ckpt.step.assign_add(1)
            print('-------------------- EPOCH ' + str(i) + ' END ------------------------')
        print('Training of SING Model done!')
