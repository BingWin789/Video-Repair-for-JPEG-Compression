import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from params import trainargs

_FILE_PATTERN = '%s-*'

class Dataset(object):
    def __init__(self,  
                dataset_name,
                dataset_dir,
                image_prefix,
                batch_size,
                patch_size, 
                num_readers=1,
                is_training=False,
                should_shuffle=False,
                should_repeat=False):
        '''
        dataset_name: train\val\test
        '''
        self.split_name = dataset_name
        self.dataset_dir = dataset_dir
        self.image_prefix = os.path.join(image_prefix, '')
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_readers = num_readers
        self.is_training = is_training
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat

    def _parse_function(self, example_proto):
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))
        
        features = {
            '1': tf.FixedLenFeature((),  tf.string, default_value=''),
            '2': tf.FixedLenFeature((),  tf.string, default_value=''),
            '3': tf.FixedLenFeature((),  tf.string, default_value=''),
            '4': tf.FixedLenFeature((),  tf.string, default_value=''),
            '5': tf.FixedLenFeature((),  tf.string, default_value=''),
            '6': tf.FixedLenFeature((),  tf.string, default_value=''),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        samples = {'prev2': tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(self.image_prefix + parsed_features['1']), channels=3), tf.float32),
                   'prev1': tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(self.image_prefix + parsed_features['2']), channels=3), tf.float32),
                   'curr': tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(self.image_prefix + parsed_features['3']), channels=3), tf.float32),
                   'next1': tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(self.image_prefix + parsed_features['4']), channels=3), tf.float32),
                   'next2': tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(self.image_prefix + parsed_features['5']), channels=3), tf.float32),
                   'label': tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(self.image_prefix + parsed_features['6']), channels=3), tf.float32)}

        return samples

    def _preprocess_image(self, sample):        
        prev2_inp = sample['prev2']
        prev1_inp = sample['prev1']
        curr_inp = sample['curr']
        next1_inp = sample['next1']
        next2_inp = sample['next2']

        lbl = sample['label']

        border = tf.py_func(calc_black_border_height, [curr_inp], tf.int32)

        shape = tf.shape(curr_inp)

        up = tf.random_uniform(
            shape=[],
            minval=border,
            maxval=shape[0] - self.patch_size - border,
            dtype=tf.int32)

        left = tf.random_uniform(
            shape=[],
            minval=0,
            maxval=shape[1] - self.patch_size,
            dtype=tf.int32)

        prev2_inp = tf.slice(prev2_inp, [up, left, 0],
                    [self.patch_size, self.patch_size, -1])
        prev1_inp = tf.slice(prev1_inp, [up, left, 0],
                    [self.patch_size, self.patch_size, -1])
        curr_inp = tf.slice(curr_inp, [up, left, 0],
                    [self.patch_size, self.patch_size, -1])
        next1_inp = tf.slice(next1_inp, [up, left, 0],
                    [self.patch_size, self.patch_size, -1])
        next2_inp = tf.slice(next2_inp, [up, left, 0],
                    [self.patch_size, self.patch_size, -1])


        lbl = tf.slice(lbl, [up, left, 0],
                     [self.patch_size, self.patch_size, -1])

        def _to_be_or_not_to_be(values, fn):
            def _to_be():
                return [fn(v) for v in values]

            def _not_to_be():
                return values

            pred = tf.less(
                tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32),
                0.5)
            values = tf.cond(pred, _to_be, _not_to_be)
            return values

        prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp, lbl = _to_be_or_not_to_be([prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp, lbl], tf.image.flip_left_right)
        prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp, lbl = _to_be_or_not_to_be([prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp, lbl], tf.image.flip_up_down)
        prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp, lbl = _to_be_or_not_to_be([prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp, lbl], tf.image.rot90)

        prev2_inp.set_shape([self.patch_size, self.patch_size, 3])
        prev1_inp.set_shape([self.patch_size, self.patch_size, 3])
        curr_inp.set_shape([self.patch_size, self.patch_size, 3])
        next1_inp.set_shape([self.patch_size, self.patch_size, 3])
        next2_inp.set_shape([self.patch_size, self.patch_size, 3])

        lbl.set_shape([self.patch_size, self.patch_size, 3])

        inputs = tf.stack([prev2_inp, prev1_inp, curr_inp, next1_inp, next2_inp], axis=0)
        inputs.set_shape([5, self.patch_size, self.patch_size, 3])

        return {'input': inputs, 'label': lbl}
        
    def get_one_shot_iterator(self):
        """Gets an iterator that iterates across the dataset once.
        Returns:
        An iterator of type tf.data.Iterator.
        """
        files = self._get_all_files()

        dataset = (
            tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
            .map(self._parse_function, num_parallel_calls=self.num_readers)
            .map(self._preprocess_image, num_parallel_calls=self.num_readers))

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat()  # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset.make_one_shot_iterator()

    def _get_all_files(self):
        """Gets all the files to read data from.
        Returns:
        A list of input files.
        """
        file_pattern = _FILE_PATTERN
        file_pattern = os.path.join(self.dataset_dir,
                                    file_pattern % self.split_name)
        return tf.gfile.Glob(file_pattern)


def calc_black_border_height(img):
    """calculate black border height of video."""
    _img = np.uint8(img * 255)
    h, w = _img.shape[:2]
    # resize image to reduce calculation cost.
    _img = cv2.resize(_img, (int(w*0.25), int(h*0.25)))
    gry = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)  # TODO: rgb in tf

    # use otsu binary segmentation to find black border.
    _, msk = cv2.threshold(gry, 0, 1, cv2.THRESH_OTSU)
    h, w = gry.shape[:2]
    ratio = 0.05
    # after otsu, black border is marked by 0,
    # if number of 1 in one line of image is less than thres,
    # this line is black border.
    # use binary search to find black border boundary line fastly.
    thres = w * ratio 
    beg, end = 0, h // 2
    while beg < end:
        cntr = (beg + end) // 2
        if np.sum(msk[cntr, :]) > thres:
            end = cntr - 1
        else:
            beg = cntr + 1
    upbeg, upend = 0, cntr

    beg, end = h // 2, h - 1
    while beg < end:
        cntr = (beg + end) // 2
        if np.sum(msk[cntr, :]) > thres:
            beg = cntr + 1
        else:
            end = cntr - 1
    dnbeg, dnend = cntr, h - 1

    uplen = upend + 1
    dnlen = dnend - dnbeg + 1
    blkbrdr = min(uplen, dnlen)
    blkbrdr *= 1/0.25  # resize back.

    if blkbrdr < 5:
        # return 0
        blkbrdr = 0
    if blkbrdr >= (h - trainargs.patch_size) // 2:
        blkbrdr = (h - trainargs.patch_size) // 2 - 3
    
    return np.int32(blkbrdr)