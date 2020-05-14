import tensorflow as tf
import tensorflow.train as tr

def process_frame_tf():
    # read from serialized strings to extract waymo data stored as tfrecord format
    tfrecord_f = '/home/serene/Documents/waymo/training_0000/'
    raw_data = tf.data.TFRecordDataset(tfrecord_f)

    return raw_data

