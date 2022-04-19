import chex
import numpy as np

class DatasetIterator:
    """Create an iterator that returns batches of data. This is useful for iterating through
    a dataset and performing multiple forward passes without overloading the GPU."""
    def __init__(self, batch_size: int, dataset: chex.Array):
        self.batch_size = batch_size
        self.n_splits = int(np.ceil(dataset.shape[0] / batch_size))  # roundup
        self.dataset_iter = iter(np.array_split(dataset, self.n_splits))
        self.n_points = dataset.shape[0]

    def __next__(self):
        return next(self.dataset_iter)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_splits