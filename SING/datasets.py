import tensorflow as tf

# Prepare NSynth TFRecordDataset to pass to model
def prepare_nsynth_dataset(dataset, batch_size = 64, use_time_embed = True):
    def process_parsed_features(point):
        input_dict = {}
        input_dict['note'] = point['note']
        input_dict['note_str'] = point['note_str']
        input_dict['instrument'] = point['instrument']
        input_dict['instrument_str'] = point['instrument_str']
        input_dict['pitch'] = point['pitch']
        input_dict['velocity'] = point['velocity']
        input_dict['instrument_family'] = point['instrument_family']
        input_dict['instrument_family_str'] = point['instrument_family_str']
        input_dict['instrument_source'] = point['instrument_source']
        input_dict['instrument_source_str'] = point['instrument_source_str']
        return {'inputs': input_dict, 'outputs': point['audio']}

    def parse_nsynth(example_proto):
        features = {
            "audio": tf.io.FixedLenFeature((4 * 16000), tf.float32),
            "note": tf.io.FixedLenFeature((), dtype = tf.int64),
            "note_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "instrument": tf.io.FixedLenFeature((), dtype = tf.int64),
            "instrument_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "instrument_source": tf.io.FixedLenFeature((), dtype = tf.int64),
            "instrument_source_str": tf.io.FixedLenFeature((), dtype = tf.string),
            "instrument_family": tf.io.FixedLenFeature((), dtype = tf.int64),
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

    return tfr_dataset_eager(dataset, batch_size)