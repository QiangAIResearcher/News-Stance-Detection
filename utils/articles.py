import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes


class Articles(object):
    def __init__(self,headlines,bodies, labels,one_hot=False,dtype=dtypes.float32):
        """Construct a DataSet. `dtype` can be either`uint8` to leave the input as `[0, 255]`, or `float32` to rescale into`[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid text dtype %r, expected uint8 or float32' %
                            dtype)

        assert headlines.shape[0] ==bodies.shape[0] == labels.shape[0], (
                'headlines.shape: %s bodies.shape: %s labels.shape: %s' % (headlines.shape, bodies.shape, labels.shape))
        self._num_examples = headlines.shape[0]

        self._headlines = np.asarray(headlines)
        self._bodies = np.asarray(bodies)
        self._input = np.column_stack((self._headlines, self._bodies))
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def headlines(self):
        return self._headlines

    @property
    def bodies(self):
        return self._bodies

    @property
    def distances(self):
        return self._distances

    @property
    def input(self):
        return self._input

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size=50):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._headlines = self._headlines[perm]
          self._bodies = self._bodies[perm]
          self._input = self._input[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._input[start:end], self._labels[start:end]

    def next_batch_2d(self, batch_size=50):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._headlines = self._headlines[perm]
          self._bodies = self._bodies[perm]
          self._input = self._input[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._headlines[start:end], self._bodies[start:end], self._labels[start:end]
