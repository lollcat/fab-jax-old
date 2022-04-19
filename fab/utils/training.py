import chex
import numpy as np

class DatasetIterator:
    """Create an iterator that returns batches of data. This is useful for iterating through
    a dataset and performing multiple forward passes without overloading the GPU."""
    def __init__(self, batch_size: int, dataset: chex.Array):
        self.batch_size = batch_size
        self.n_splits = int(np.ceil(dataset.shape[0] / batch_size))  # roundup
        self.dataset = dataset
        self.dataset_iter = iter(np.array_split(self.dataset, self.n_splits))
        self.n_points = dataset.shape[0]
        self.count = 0

    def __next__(self):
        result = next(self.dataset_iter)
        self.count += 1
        if self.count == self.__len__():
            self.count = 0
            self.dataset_iter = iter(np.array_split(self.dataset, self.n_splits))
            raise StopIteration
        return result

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_splits