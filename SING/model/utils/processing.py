import os
import tensorflow as tf
import IPython
from matplotlib import pyplot as plt

def get_tensorflow_checkpoint(optimizer, model_checkpoint_dir, max_to_keep = 3):
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer, net = self)
    manager = tf.train.CheckpointManager(
        ckpt,
        model_checkpoint_dir,
        max_to_keep = max_to_keep
    )
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing model from scratch.")
    return ckpt, manager


def log_statistics_to_console(loss):
    print('Printing to console training statistics -')
    print(loss)
    print('Printing training statistics done!')

def get_all_trainable_variables(trainable_sub_models = []):
    trainable_variables = []
    for model in trainable_sub_models:
        for var in model.trainable_variables:
            trainable_variables.append(var)
    return trainable_variables

def log_training_audio_to_notebook(real_wav, generated_wav, num_outputs = 1, audio_sampling_rate = 16000)
    print('------------- Generating audio ---------------')
    print('Original audio waveforms - ')
    for j in range(num_samples):
        IPython.display.display(
            IPython.display.Audio(
                real_wav[i,:],
                rate = audio_sampling_rate
            )
        )

    print('Generated audio waveforms -')
    for j in range(num_samples):
        IPython.display.display(
            IPython.display.Audio(
                generated_wav[i,:],
                rate = audio_sampling_rate
            )
        )
    print('--------------- Generated audio! ---------------')
    print