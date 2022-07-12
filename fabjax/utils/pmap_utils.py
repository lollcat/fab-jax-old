import jax

def get_from_first_device(nest, as_numpy: bool = True):
    # copied from https://github.com/deepmind/acme/blob/master/acme/jax/utils.py
  """Gets the first array of a nest of `jax.pxla.ShardedDeviceArray`s.
  Args:
    nest: A nest of `jax.pxla.ShardedDeviceArray`s.
    as_numpy: If `True` then each `DeviceArray` that is retrieved is transformed
      (and copied if not on the host machine) into a `np.ndarray`.
  Returns:
    The first array of a nest of `jax.pxla.ShardedDeviceArray`s. Note that if
    `as_numpy=False` then the array will be a `DeviceArray` (which will live on
    the same device as the sharded device array). If `as_numpy=True` then the
    array will be copied to the host machine and converted into a `np.ndarray`.
  """
  zeroth_nest = jax.tree_map(lambda x: x[0], nest)
  return jax.device_get(zeroth_nest) if as_numpy else zeroth_nest