import h5py
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial


def load_shd_sparse(file_path):
    """
    Loads spike data from an SHD HDF5 file but keeps it in a sparse format
    to save memory. It returns lists of arrays, where each element corresponds
    to a sample.

    The expected HDF5 structure is:
    /
    ├── labels (Dataset, shape: (n_samples,))
    └── spikes (Group)
        ├── times (Dataset, shape: (n_samples,), dtype: object)
        │     -> each element is a np.array of spike timestamps
        └── units (Dataset, shape: (n_samples,), dtype: object)
              -> each element is a np.array of corresponding neuron IDs
    """
    with h5py.File(file_path, "r") as f:
        if "spikes" not in f or "labels" not in f:
            raise ValueError("HDF5 file must contain 'spikes' and 'labels'.")

        labels = jnp.array(f["labels"][:])

        spike_times = [arr[:] for arr in f["spikes"]["times"]]
        spike_units = [arr[:] for arr in f["spikes"]["units"]]

    return spike_times, spike_units, labels


@partial(jit, static_argnums=(3, 4, 5))
def _convert_to_dense_jit(
    sample_indices, time_bins, units, num_samples, n_time_steps, n_features
):
    """JIT-compiled function to create a dense tensor from sparse data."""
    dense_tensor = jnp.zeros((num_samples, n_time_steps, n_features), dtype=jnp.bool_)
    dense_tensor = dense_tensor.at[sample_indices, time_bins, units].set(True)
    return dense_tensor


def convert_to_dense_tensor(times, units, n_time_steps, n_features=700):
    """
    Converts a single sparse spike sample (times and units) into a dense
    boolean JAX array (spike train).
    """
    max_time = 1000.0
    time_bin_size = (max_time / n_time_steps) + 1e-9

    dense_tensor = jnp.zeros((n_time_steps, n_features), dtype=jnp.bool_)

    if times.size > 0:
        times_ms = times * 1000.0
        time_bins = (times_ms / time_bin_size).astype(jnp.int32)
        time_bins = jnp.clip(time_bins, 0, n_time_steps - 1)
        dense_tensor = dense_tensor.at[time_bins, units].set(True)

    return dense_tensor


def preprocess_to_dense(sparse_times, sparse_units, n_steps, n_features=700):
    """
    Converts all sparse samples to a single dense tensor using vectorized JAX operations.
    """
    num_samples = len(sparse_times)
    if num_samples == 0:
        return jnp.empty((0, n_steps, n_features), dtype=jnp.bool_)

    # Faster to pre-process with NumPy for varied-length lists
    sample_indices = np.concatenate(
        [np.full(len(t), i, dtype=np.int32) for i, t in enumerate(sparse_times)]
    )
    all_times = np.concatenate(sparse_times)
    all_units = np.concatenate(sparse_units).astype(np.int32)

    # Vectorized conversion of times to time bins
    max_time = 1000.0
    time_bin_size = (max_time / n_steps) + 1e-9
    times_ms = all_times * 1000.0
    all_time_bins = (times_ms / time_bin_size).astype(np.int32)
    all_time_bins = np.clip(all_time_bins, 0, n_steps - 1)

    # Use JIT-compiled JAX function for the final scatter operation on GPU
    return _convert_to_dense_jit(
        jnp.asarray(sample_indices),
        jnp.asarray(all_time_bins),
        jnp.asarray(all_units),
        num_samples,
        n_steps,
        n_features,
    )


def load_preprocessed(n_steps=100):
    X_train_times, X_train_units, y_train_labels = load_shd_sparse("shd/shd_train.h5")

    print("Preprocessing data...")
    X_train_dense = preprocess_to_dense(X_train_times, X_train_units, n_steps)
    print("Preprocessing complete.")

    X_test_times, X_test_units, y_test_labels = load_shd_sparse("shd/shd_test.h5")
    print("Preprocessing test data...")
    X_test_dense = preprocess_to_dense(X_test_times, X_test_units, n_steps)
    print("Test data preprocessing complete.")

    return X_train_dense, y_train_labels, X_test_dense, y_test_labels
