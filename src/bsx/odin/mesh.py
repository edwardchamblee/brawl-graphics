"""Decode Odin interleaved vertex streams into glTF-style attribute arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .formats import ATTRIBUTE_SEMANTICS, AttributeFormat, format_info

# Weight of each packed bone-weight bit field (matches the game's shader constant).
_PACKED_WEIGHT_SCALE = 0.0002442


def vertex_capacity(view_length: int, info: dict[str, Any]) -> int:
    """Largest vertex count that fits every stream of a meshDataInfo."""
    capacity = None
    for descriptor in info.get("vertexDescriptors", []):
        stride = int(descriptor.get("stride", 0))
        offset = int(descriptor.get("offset", 0))
        if stride <= 0:
            continue
        available = max(0, view_length - offset) // stride
        capacity = available if capacity is None else min(capacity, available)
    return capacity or 0


def _read_stream(
    view: memoryview, offset: int, stride: int, byte_size: int, count: int
) -> np.ndarray:
    raw = np.frombuffer(view, dtype=np.uint8)
    if count == 0:
        return np.zeros((0, byte_size), dtype=np.uint8)
    needed = offset + (count - 1) * stride + byte_size
    if needed > len(raw):
        raise ValueError("vertex stream exceeds buffer view")
    rows = np.lib.stride_tricks.as_strided(
        raw[offset:], shape=(count, byte_size), strides=(stride, 1)
    )
    return np.ascontiguousarray(rows)


def _unpack_bone_weights(packed: np.ndarray) -> np.ndarray:
    values = packed.reshape(-1).astype(np.uint32)
    x = (values >> 21).astype(np.float32) * _PACKED_WEIGHT_SCALE
    y = ((values >> 10) & 0x7FF).astype(np.float32) * _PACKED_WEIGHT_SCALE
    z = (values & 0x3FF).astype(np.float32) * _PACKED_WEIGHT_SCALE
    w = 1.0 - x - y - z
    return np.stack([w, x, y, z], axis=1).astype(np.float32)


def _unpack_1010102(packed: np.ndarray, signed: bool) -> np.ndarray:
    values = packed.reshape(-1).astype(np.uint32)
    parts = [(values & 0x3FF), (values >> 10) & 0x3FF, (values >> 20) & 0x3FF]
    alpha = (values >> 30) & 0x3
    if signed:
        comps = [((p.astype(np.int32) ^ 0x200) - 0x200).astype(np.float32) / 511.0 for p in parts]
        comps.append(((alpha.astype(np.int32) ^ 0x2) - 0x2).astype(np.float32))
    else:
        comps = [p.astype(np.float32) / 1023.0 for p in parts]
        comps.append(alpha.astype(np.float32) / 3.0)
    return np.stack(comps, axis=1)


def _to_float(array: np.ndarray, normalized: bool) -> np.ndarray:
    if array.dtype.kind == "f":
        return array.astype(np.float32)
    if not normalized:
        return array.astype(np.float32)
    info = np.iinfo(array.dtype)
    result = array.astype(np.float32) / float(info.max)
    if info.min < 0:
        result = np.maximum(result, -1.0)
    return result


def decode_attribute(raw: np.ndarray, name: str, fmt: AttributeFormat) -> np.ndarray:
    """Convert one raw interleaved attribute into its glTF representation."""
    info = format_info(fmt)
    typed = raw.view(info.dtype).reshape(raw.shape[0], info.components)

    if name == "a_boneweights" and fmt == AttributeFormat.UInt:
        return _unpack_bone_weights(typed)
    if fmt == AttributeFormat.UInt1010102Norm:
        typed = _unpack_1010102(typed, signed=False)
    elif fmt == AttributeFormat.Int1010102Norm:
        typed = _unpack_1010102(typed, signed=True)

    semantic = ATTRIBUTE_SEMANTICS.get(name)

    if semantic == "JOINTS_0":
        if typed.dtype in (np.uint8, np.uint16):
            joints = typed
        else:
            joints = typed.astype(np.uint16)
        return _fit_components(joints, 4, fill=0)

    values = _to_float(typed, info.normalized)

    if semantic == "POSITION":
        return _fit_components(values, 3, fill=0.0)
    if semantic == "NORMAL":
        normals = _fit_components(values, 3, fill=0.0)
        return _normalize_rows(normals)
    if semantic == "TANGENT":
        tangents = _fit_components(values, 4, fill=1.0)
        tangents[:, :3] = _normalize_rows(tangents[:, :3])
        return tangents
    if semantic and semantic.startswith("TEXCOORD"):
        return _fit_components(values, 2, fill=0.0)
    if semantic and semantic.startswith("COLOR"):
        return _fit_components(values, 4, fill=1.0)
    if semantic == "WEIGHTS_0":
        return _fit_components(values, 4, fill=0.0)
    return values


def _fit_components(array: np.ndarray, components: int, fill: float) -> np.ndarray:
    if array.shape[1] == components:
        return np.ascontiguousarray(array)
    if array.shape[1] > components:
        return np.ascontiguousarray(array[:, :components])
    padded = np.full((array.shape[0], components), fill, dtype=array.dtype)
    padded[:, : array.shape[1]] = array
    return padded


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (vectors / lengths).astype(np.float32)


def decode_mesh_data_info(
    view: memoryview, info: dict[str, Any], count: int
) -> dict[str, np.ndarray]:
    """Decode ``count`` vertices of a meshDataInfo into ``{semantic: array}``.

    Attributes without a glTF semantic (``a_colorMul``, ``a_model``...) are
    returned under their raw Odin name prefixed with an underscore so they
    survive as custom attributes.
    """
    attributes: dict[str, np.ndarray] = {}
    for descriptor in info.get("vertexDescriptors", []):
        base = int(descriptor.get("offset", 0))
        stride = int(descriptor.get("stride", 0))
        for attribute in descriptor.get("attributes", []):
            name = str(attribute.get("name"))
            fmt = AttributeFormat(int(attribute.get("format")))
            fmt_info = format_info(fmt)
            raw = _read_stream(
                view, base + int(attribute.get("offset", 0)), stride, fmt_info.byte_size, count
            )
            decoded = decode_attribute(raw, name, fmt)
            semantic = ATTRIBUTE_SEMANTICS.get(name, f"_{name.upper()}")
            attributes[semantic] = decoded
    return attributes
