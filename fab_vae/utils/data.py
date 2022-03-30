from typing import Iterator, Mapping, Sequence

import chex
import numpy as np

import tensorflow_datasets as tfds

Batch = Mapping[str, np.ndarray]
MNIST_IMAGE_SHAPE: chex.Shape = (28, 28, 1)

def load_dataset(split: str, batch_size: int) -> Iterator[Batch]:
  ds = tfds.load("binarized_mnist", split=split, shuffle_files=True)
  ds = ds.shuffle(buffer_size=10 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))