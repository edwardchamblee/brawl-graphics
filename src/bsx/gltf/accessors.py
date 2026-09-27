"""Read standard glTF accessors and build new binary buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

COMPONENT_DTYPES: dict[int, np.dtype] = {
    5120: np.dtype(np.int8),
    5121: np.dtype(np.uint8),
    5122: np.dtype(np.int16),
    5123: np.dtype(np.uint16),
    5125: np.dtype(np.uint32),
    5126: np.dtype(np.float32),
}

DTYPE_COMPONENTS: dict[np.dtype, int] = {v: k for k, v in COMPONENT_DTYPES.items()}

TYPE_COMPONENT_COUNT: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def component_type_of(value: int) -> int:
    """Strip Supercell's private flag bits from an accessor componentType."""
    return int(value) & 0xFFFF


def buffer_view_bytes(gltf: dict[str, Any], binary: bytes, index: int) -> memoryview:
    view = gltf["bufferViews"][index]
    if view.get("buffer", 0) != 0:
        raise ValueError("only single-buffer GLB files are supported")
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    if end > len(binary):
        raise ValueError(f"bufferView {index} exceeds BIN chunk")
    return memoryview(binary)[start:end]


def read_accessor(gltf: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    """Return accessor ``index`` as an array of shape ``(count, components)``."""
    accessor = gltf["accessors"][index]
    if "sparse" in accessor:
        raise NotImplementedError("sparse accessors are not supported")

    dtype = COMPONENT_DTYPES[component_type_of(accessor["componentType"])]
    components = TYPE_COMPONENT_COUNT[accessor["type"]]
    count = accessor["count"]

    if "bufferView" not in accessor:
        return np.zeros((count, components), dtype=dtype)

    view = gltf["bufferViews"][accessor["bufferView"]]
    data = buffer_view_bytes(gltf, binary, accessor["bufferView"])
    offset = accessor.get("byteOffset", 0)
    element_size = dtype.itemsize * components
    stride = view.get("byteStride") or element_size

    if stride == element_size:
        array = np.frombuffer(data, dtype=dtype, count=count * components, offset=offset)
        return array.reshape(count, components)

    rows = np.lib.stride_tricks.as_strided(
        np.frombuffer(data, dtype=np.uint8, offset=offset),
        shape=(count, element_size),
        strides=(stride, 1),
    )
    return np.ascontiguousarray(rows).view(dtype).reshape(count, components)


@dataclass
class BufferBuilder:
    """Accumulates binary data and emits bufferViews/accessors for it."""

    data: bytearray = field(default_factory=bytearray)
    buffer_views: list[dict[str, Any]] = field(default_factory=list)
    accessors: list[dict[str, Any]] = field(default_factory=list)

    def _align(self, alignment: int = 4) -> None:
        remainder = len(self.data) % alignment
        if remainder:
            self.data += b"\x00" * (alignment - remainder)

    def add_buffer_view(self, payload: bytes, target: int | None = None) -> int:
        self._align()
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": len(self.data),
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        self.data += payload
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        array: np.ndarray,
        *,
        target: int | None = None,
        normalized: bool = False,
        min_max: bool = False,
    ) -> int:
        array = np.ascontiguousarray(array)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("accessor arrays must be 1D or 2D")

        count, components = array.shape
        type_name = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 9: "MAT3", 16: "MAT4"}[
            components
        ]
        dtype = array.dtype
        if dtype not in DTYPE_COMPONENTS:
            raise ValueError(f"unsupported accessor dtype {dtype}")

        view_index = self.add_buffer_view(array.tobytes(), target)
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": DTYPE_COMPONENTS[dtype],
            "count": int(count),
            "type": type_name,
        }
        if normalized:
            accessor["normalized"] = True
        if min_max and count:
            accessor["min"] = [float(v) for v in array.min(axis=0)]
            accessor["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def finish(self) -> bytes:
        self._align()
        return bytes(self.data)
