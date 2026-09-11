"""Bounded, instance-owned access to immutable HDF5 spin slices."""

# 2026-09-11 15:42 CST (linux): Each read owns a contiguous slice, never a view
# pinning an entire region. No HDF5 handle or global device pool is retained.
import operator
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


def file_identity(filename):
    stat = Path(filename).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class SpinSliceCache(Sequence):
    """Read-only sequence with a per-region LRU, measured in spin slices.

    Factories must not retain the parent amplitude object (avoid ownership
    cycles), and must consume independent FP64 host arrays. The original FEW
    native interpolation synchronizes before returning; caches are not shared
    between concurrent calls. File replacement/modification is rejected on miss.
    """

    def __init__(self, filename, dataset, indices, factory, capacity, identity):
        self.filename = Path(filename).resolve()
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.factory = factory
        self.capacity = capacity
        self._identity = identity
        self._cache = OrderedDict()
        self.loads = self.hits = self.evictions = self.bytes_read = 0

    def _stat(self):
        return file_identity(self.filename)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Spin slice index out of range")
        physical = self.indices[index]
        if physical in self._cache:
            self.hits += 1
            self._cache.move_to_end(physical)
            return self._cache[physical]
        if self._stat() != self._identity:
            raise RuntimeError("Amplitude table changed after lazy construction")
        with h5py.File(self.filename, "r") as file:
            # HDF5 hyperslab selection reads this slice, not the entire region.
            coeff = file[self.dataset][physical, ...]
        if self._stat() != self._identity:
            raise RuntimeError("Amplitude table changed during lazy read")
        if coeff.dtype != np.float64 or not coeff.flags.c_contiguous:
            raise ValueError("Native amplitude coefficients must be contiguous FP64")
        value = self.factory(coeff)
        # Keep the source array alive through any asynchronous host-to-device
        # upload. This fence only runs on misses, not on repeated cache hits.
        if value.backend.uses_cuda:
            value.xp.cuda.get_current_stream().synchronize()
        self._cache[physical] = value
        self.loads += 1
        self.bytes_read += coeff.nbytes
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
            self.evictions += 1
        return value

    def clear(self):
        """Release cached objects, not the backend's shared allocation pool."""
        self._cache.clear()

    def prepare(self, indices):
        # 2026-09-11 15:49 CST (linux): Protect already-resident nodes needed
        # by this pair before upper-first evaluation inserts a new upper node.
        for index in indices:
            physical = self.indices[operator.index(index)]
            if physical in self._cache:
                self._cache.move_to_end(physical)

    def cache_info(self):
        """Logical coefficients, not process RSS or GPU allocator reservations."""
        return dict(
            capacity=self.capacity,
            resident_indices=list(self._cache),
            loads=self.loads,
            hits=self.hits,
            evictions=self.evictions,
            coefficient_bytes=sum(value.coeff.nbytes for value in self._cache.values()),
            bytes_read=self.bytes_read,
        )
