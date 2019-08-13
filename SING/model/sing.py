import os
import tensorflow as tf
import numpy as np
from .conv_ae import ConvolutionalEncoder, ConvolutionalDecoder
from .sequence_encoder import SequenceEncoder
from .processing import *
from .losses import *
from .tensorboard import *

class SINGModel(tf.keras.Model):
    def __init__(self, model_log_dir = 'model_logs_sing'):
        super(SINGModel, self).__init__(name = "SINGModel")
        # Model variables
        self.conv_encoder = ConvolutionalEncoder()
        self.conv_encoder.build(input_shape=(10,64000,1))
        print(self.conv_encoder.summary())
        self.conv_decoder = ConvolutionalDecoder()
        self.conv_decoder.build(input_shape=(10,250,128))
        print(self.conv_decoder.summary())
        self.sequence_generator = SequenceEncoder()

        # Extra variables
        self.learning_rate = 0.0003
        self.num_epochs = 50
        self.model_log_dir = model_log_dir
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.num_steps_checkpoint = 1000
        self.num_outputs = 3
        self.sampling_rate = 16000
        # This means that training starts at step 0
        # Autoencoder is trained till step 100000 (dataset_size * num_epochs)
        # LSTM is trained till 200000
        # LSTM is trained with conv_decoder till 300000
        self.max_steps_ae = 220000
        self.max_steps_lstm = 260000
        self.max_steps_final = 300000

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
    def train(self, dataset, train_phase = 'full_pipeline'):

        def pretrain_sequence_generator(data):
            with tf.GradientTape() as tape:
                encoding = self.sequence_generator(
                    tf.cast(data['inputs']['instrument'], tf.float32),
                    tf.cast(data['inputs']['pitch'], tf.float32),
                    tf.cast(data['inputs']['velocity'], tf.float32),
                    250
                )
                encoder_output = self.conv_encoder(tf.expand_dims(data['outputs'], axis = -1))
                loss = tf.keras.losses.MSE(encoder_output, encoding)
            trainable_variables = self.sequence_generator.trainable_variables
            grads = tape.gradient(loss, trainable_variables)
            grads_and_vars = zip(grads, trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)
            return encoding, loss, grads
            
        def pretrain_step_conv_autoencoder(waveform, loss_type = 'MSE'):        
            with tf.GradientTape() as tape:
                encoding = self.conv_encoder(waveform)
                decoding = self.conv_decoder(encoding)
                target_wav = tf.squeeze(waveform, axis = 2)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(target_wav, decoding)
                else:
                    loss = spectral_loss(data['outputs'], decoding)
            trainable_sub_models = [self.conv_encoder, self.conv_decoder]
            trainable_variables = get_all_trainable_variables(trainable_sub_models)
            grads = tape.gradient(loss, trainable_variables)
            grads_and_vars = zip(grads, trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)
            return decoding, loss, grads

        def train_step_sing(data, loss_type = 'MSE'):
            with tf.GradientTape() as tape:
                encoding = self.sequence_generator(data['inputs']['instrument'], data['inputs']['pitch'], data['inputs']['velocity'], 250)
                decoding = self.conv_decoder(encoding)
                if loss_type == 'MSE':
                    loss = tf.keras.losses.MSE(data['outputs'], decoding)
                else:
                    loss = spectral_loss(data['outputs'], decoding)
                trainable_sub_models = [self.sequence_generator, self.conv_decoder]
                trainable_variables = get_all_trainable_variables(trainable_sub_models)
                grads = tape.gradient(loss, trainable_variables)
                grads_and_vars = zip(grads, trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)
            return decoding, loss, grads

        run_full_pipeline = train_phase == 'full_pipeline'
        run_autoenc_pretrain = train_phase == 'pretrain_autoenc' or run_full_pipeline
        run_lstm_pretrain = train_phase == 'pretrain_lstm' or run_full_pipeline
        run_fine_tuning = train_phase == 'fine_tuning' or run_full_pipeline
        print('Lets start training. Currently the training phase that the model is running in ...')
        if run_full_pipeline:
            print('Pretrain autoencoder -> Pretrain sequence generator LSTM -> Fine tune both models')
        elif run_autoenc_pretrain:
            print('******** Pretraining AutoEncoder ********')
        elif run_lstm_pretrain:
            print('******** Pretrain Sequence Generator LSTM ********')
        elif run_fine_tuning:
            print('******** Fine tune AutoEncoder and Sequence Generator LSTM models ********')
        else:
            print('******** No training will be taking place ********')

        # Pre-Training Conv Autoencoder
        if run_autoenc_pretrain:
            ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = self.optimizer, net = self)
            manager = get_tensorflow_checkpoint(
                ckpt,
                self.optimizer,
                self.model_log_dir
            )
            print('Initializing pretraining of Convolutional Autoencoder ...')
            for i in range(self.num_epochs):
                print('-------------------- EPOCH ' + str(i) + ' ------------------------')
                for data in dataset:
                    step = int(ckpt.step)
                    if step > self.max_steps_ae:
                        break
                    output_wav, loss, grads = pretrain_step_conv_autoencoder(
                        tf.expand_dims(data['outputs'], axis = -1)
                    )
                    log_stuff_to_tensorboard(
                        step,
                        tf.reduce_sum(loss),
                        grads
                    )
                    if step % self.num_steps_checkpoint == 0:
                        print("============== STEP " + str(step) + " ==============")
                        log_statistics_to_console(
                            tf.reduce_sum(loss)
                        )
                        log_training_audio_to_notebook(
                            data,
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

        # Pre-Training LSTM Sequence Generator
        if run_lstm_pretrain:    
            ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = self.optimizer, net = self)
            manager = get_tensorflow_checkpoint(
                ckpt,
                self.optimizer,
                self.model_log_dir
            )
            print('Initializing pretraining of LSTM Sequence Generator ...')
            for i in range(self.num_epochs):
                print('-------------------- EPOCH ' + str(i) + ' ------------------------')
                for data in dataset:
                    step = int(ckpt.step)
                    if step > self.max_steps_lstm:
                        break
                    output_wav, loss, grads = pretrain_sequence_generator(data)
                    log_stuff_to_tensorboard(
                        step,
                        tf.reduce_sum(loss),
                        grads
                    )
                    if step % self.num_steps_checkpoint == 0:
                        print("============== STEP " + str(step) + " ==============")
                        log_statistics_to_console(
                            tf.reduce_sum(loss)
                        )
                        save_path = manager.save()
                        print("Saved checkpoint for step {}: {}".format(step, save_path))
                        print("============== STEP END ==============")
                    ckpt.step.assign_add(1)
                print('-------------------- EPOCH ' + str(i) + ' END ------------------------')
            print('Pretraining of LSTM Sequence Generator done!')

        # Final training of entire model
        if run_fine_tuning:
            print('Initializing training of SING Model with LSTM sequence generator ...')
            ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = self.optimizer, net = self)
            manager = get_tensorflow_checkpoint(
                ckpt,
                self.optimizer,
                self.model_log_dir
            )
            for i in range(self.num_epochs):
                print('-------------------- EPOCH ' + str(i) + ' ------------------------')
                for data in dataset:
                    step = int(ckpt.step)
                    if step > self.max_steps_final:
                        break
                    output_wav, loss, grads = train_step_sing(data)
                    log_stuff_to_tensorboard(
                        step,
                        tf.reduce_sum(loss),
                        grads
                    )
                    if step % self.num_steps_checkpoint == 0:
                        print("============== STEP " + str(step) + " ==============")
                        log_statistics_to_console(
                            tf.reduce_sum(loss)
                        )
                        log_training_audio_to_notebook(
                            data,
                            output_wav,
                            num_outputs = self.num_outputs,
                            audio_sampling_rate = self.sampling_rate
                        )
                        save_path = manager.save()
                        print("Saved checkpoint for step {}: {}".format(step, save_path))
                        print("============== STEP END ==============")
                    ckpt.step.assign_add(1)
                print('-------------------- EPOCH ' + str(i) + ' END ------------------------')
            print('Fine tuning of AutoEncoder and Sequence Generator LSTM models done!')
        print('Training phase done!')