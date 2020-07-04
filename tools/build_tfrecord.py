import six
import os
import math
import sys
import tensorflow as tf
import random
import shutil
import pickle
from ..params import datasetargs

split = 'train'
_NUM_SHARDS = 10
tfresrt = datasetargs.tfrecord_dir
damage_img_prefix = datasetargs.damage_sub_dir
label_img_prefix = datasetargs.label_sub_dir
train_data_path = datasetargs.pkl_path

with open(train_data_path, 'rb') as f:
    train_data = pickle.load(f)


def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
        values: A string.
    Returns:
        A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))

def convert_to_tfexample(frames):
    a, b, c, d, e, f = frames
    return tf.train.Example(features=tf.train.Features(feature={
        '1': _bytes_list_feature(a),
        '2': _bytes_list_feature(b),
        '3': _bytes_list_feature(c),
        '4': _bytes_list_feature(d),
        '5': _bytes_list_feature(e),
        '6': _bytes_list_feature(f),
    }))

def _convert_dataset(dataset):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, test).
  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  sys.stdout.write('Processing ' + dataset)
  num_images = len(train_data)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        tfresrt,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        
        frames = [os.path.join(damage_img_prefix, j) for j in train_data[i]]
        frames.append(os.path.join(label_img_prefix, train_data[i][2]))
        
        # Convert to tf example.
        example = convert_to_tfexample(frames)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  _convert_dataset(split)


if __name__ == '__main__':
  tf.app.run()

