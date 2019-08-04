import tensorflow as tf

# Prepare NSynth TFRecordDataset to pass to model
def prepare_nsynth_dataset(dataset, time_embeddings = True):
    def process_parsed_features(point):
        instr_dim_size = 1006
        instr_embedding_size = 16
        instrument = tf.one_hot(point['instrument'], instr_dim_size)

        pitch_dim_size = 128
        pitch_embedding_size = 8
        pitch = tf.one_hot(point['pitch'], pitch_dim_size)

        vel_dim_size = 128
        vel_embedding_size = 2
        velocity = tf.one_hot(point['velocity'], vel_dim_size)

        audio = point['audio']

        time_dim_size = 250
        time_embedding_size = 4

        original_input_size = instr_dim_size + pitch_dim_size + vel_dim_size + time_dim_size
        embedding_input_size = instr_embedding_size + pitch_embedding_size + vel_embedding_size + time_embedding_size

        inputs = tf.concat([instrument, pitch, velocity], axis = 0)
        # Modification for time embeddings
        if time_embeddings:
            timesteps = 250
            temp = []
            for i in range(timesteps):
                x = inputs
                x = tf.concat([x, tf.one_hot(i, timesteps)], axis = 0)
                temp.append(inputs)
            inputs = tf.stack(temp, axis = -1)
            shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, [250, 1262])
        return {'inputs': inputs, 'outputs': audio}

    def parse_nsynth(example_proto):
        features = {
            "audio": tf.io.FixedLenFeature((4 * 16000), tf.float32),
            "note": tf.io.FixedLenFeature((), dtype = tf.int64),
            "note_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "instrument": tf.io.FixedLenFeature((), dtype = tf.int64),
            "instrument_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "instrument_source": tf.io.FixedLenFeature((), dtype = tf.int64),
            "instrument_source_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "instrument_family_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "sample_rate": tf.io.FixedLenFeature((), dtype = tf.int64),
            "velocity": tf.io.FixedLenFeature((), dtype = tf.int64),
            "pitch": tf.io.FixedLenFeature((), dtype = tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return process_parsed_features(parsed_features)

    def tfr_dataset_eager(data, batch_size):
        data = data.apply(tf.data.experimental.shuffle_and_repeat(10000))
        data = data.apply(tf.data.experimental.map_and_batch(map_func = parse_nsynth, batch_size = batch_size))
        #data = data.prefetch(1)
        return data

    return tfr_dataset_eager(dataset, 64)