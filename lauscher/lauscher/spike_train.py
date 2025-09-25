from __future__ import annotations

from os import makedirs
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

try:
    import torch
except ImportError:
    torch = None

from lauscher.abstract import Transformable, Exportable, Plottable


class SpikeTrain(Transformable, Exportable, Plottable):
    def __init__(self):
        super().__init__()
        self._data = NotImplemented

    @property
    def spike_labels(self):
        return self._data[1]

    @property
    def spike_times(self):
        return self._data[0]

    def export(self, path: str):
        makedirs(Path(path).parent, exist_ok=True)
        np.savez(path, self._data)

    def plot(self, axis: Axes):
        axis.plot(self.spike_times, self.spike_labels,
                  ls="none", marker=".", color="black")

        axis.set_xlabel("Time")
        axis.set_ylabel("Label")

    @classmethod
    def from_dense(cls, channel_time_matrix: np.ndarray,
                   sample_rate: int) -> SpikeTrain:
        spikes = np.array(np.where(channel_time_matrix.T), dtype=np.float64)
        spikes[0, :] = spikes[0, :] / sample_rate

        result = cls()
        result._data = spikes

        return result

    @classmethod
    def from_sparse(cls, spikes: np.ndarray) -> SpikeTrain:
       
        result = cls()
        result._data = np.array(spikes, dtype=np.float64)
        return result
    
    def to_torch(self, device: str = "cpu", dtype=None):
        if torch is None:
            raise RuntimeError("PyTorch not installed")
        t = torch.from_numpy(self._data)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)
    # to_torch generates 2D tensor [Time, label]

    def to_dense(
        self,
        n_steps: int,
        num_channels: int,
        max_time_sec: float = 1.0,
        *,
        device: str = "cpu",
        dtype=None,
    ):
        """
        Convert internal sparse spike events to a dense (T, N) tensor for the SNN.

        Output:
            torch.Tensor with shape (n_steps, num_channels).
            Each [t, n] is True/1 if neuron n fired in time-bin t.

        Binning rule (aligned with SHD loader & training loop):
            - total time window = max_time_sec (default 1.0 s)
            - dt = max_time_sec / n_steps
            - bin = floor(time_sec / dt), then clamp to [0, n_steps-1]

        Args:
            n_steps: number of time bins (T).
            num_channels: number of input neurons/features (N).
            max_time_sec: total time window length in seconds.
            device: target device, e.g. "cpu", "cuda", or "mps".
            dtype: output dtype. Default torch.bool (recommended). If not bool,
                   spike positions are set to 1.

        Returns:
            Dense spike tensor of shape (n_steps, num_channels).
        """
        if torch is None:
            raise RuntimeError("PyTorch not installed; to_dense() requires torch.")

        if dtype is None:
            dtype = torch.bool

        # Empty case → all zeros
        if getattr(self, "_data", NotImplemented) is NotImplemented:
            return torch.zeros((n_steps, num_channels), dtype=dtype, device=device)

        # (2, K): row 0 = times (seconds, float), row 1 = labels (channel ids)
        events = torch.from_numpy(self._data).to(device)
        if events.numel() == 0:
            return torch.zeros((n_steps, num_channels), dtype=dtype, device=device)

        times = events[0]                                 # float seconds
        labels = events[1].long().clamp_(0, num_channels - 1)

        # Time binning
        dt = (max_time_sec / float(n_steps)) + 1e-9       # avoid edge rounding
        bins = torch.floor(times / dt).long().clamp_(0, n_steps - 1)

        # Scatter to dense (T, N)
        dense = torch.zeros((n_steps, num_channels), dtype=dtype, device=device)
        if dtype == torch.bool:
            dense[bins, labels] = True
        else:
            dense[bins, labels] = 1

        return dense

