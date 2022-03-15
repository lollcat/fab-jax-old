import abc
from typing import Any, Dict, List, Mapping, Union
import pickle
import wandb
import numpy as np
import pandas as pd
import pathlib
import tree

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    # copied from Acme: https://github.com/deepmind/acme
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""



class ListLogger(Logger):
    """Manually save the data to the class in a dict. Currently only supports scalar history
    inputs."""
    def __init__(self, save: bool = True, save_path: str = "/tmp/logging_hist.pkl",
                 save_period: int = 100):
        self.save = save
        self.save_path = save_path
        if save:
            pathlib.Path(self.save_path).parent.mkdir(exist_ok=True)
        self.save_period = save_period  # how often to save the logging history
        self.history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
        self.print_warning: bool = False
        self.iter = 0

    def write(self, data: LoggingData) -> None:
        for key, value in data.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

        self.iter += 1
        if self.save and (self.iter + 1) % self.save_period == 0:
            pickle.dump(self.history, open(self.save_path, "wb")) # overwrite with latest version

    def close(self) -> None:
        if self.save:
            pickle.dump(self.history, open(self.save_path, "wb"))


class WandbLogger(Logger):
    def __init__(self, **kwargs: Any):
        self.run = wandb.init(**kwargs)
        self.iter: int = 0

    def write(self, data: Dict[str, Any]) -> None:
        self.run.log(data, step=self.iter, commit=False)
        self.iter += 1

    def close(self) -> None:
        self.run.finish()


class PandasLogger(Logger):
    def __init__(self,
                 save: bool = True,
                 save_path: str ="/tmp/logging_history.csv",
                 save_period: int = 100):
        self.save_path = save_path
        if save:
            pathlib.Path(self.save_path).parent.mkdir(exist_ok=True)
        self.save = save
        self.save_period = save_period
        self.dataframe = pd.DataFrame()
        self.iter: int = 0

    def write(self, data: Dict[str, Any]) -> None:
        self.dataframe = self.dataframe.append(data, ignore_index=True)
        self.iter += 1
        if self.save and (self.iter + 1) % self.save_period == 0:
            self.dataframe.to_csv(open(self.save_path, "w"))  # overwrite with latest version

    def close(self) -> None:
        if self.save:
            self.dataframe.to_csv(open(self.save_path, "w")) # overwrite with latest version


# copied from Acme: https://github.com/deepmind/acme
def tensor_to_numpy(value: Any):
  if hasattr(value, 'numpy'):
    return value.numpy()  # tf.Tensor (TF2).
  if hasattr(value, 'device_buffer'):
    return np.asarray(value)  # jnp.DeviceArray.
  return value


# copied from Acme: https://github.com/deepmind/acme
def to_numpy(values: Any):
  """Converts tensors in a nested structure to numpy.
  Converts tensors from TensorFlow to Numpy if needed without importing TF
  dependency.
  Args:
    values: nested structure with numpy and / or TF tensors.
  Returns:
    Same nested structure as values, but with numpy tensors.
  """
  return tree.map_structure(tensor_to_numpy, values)