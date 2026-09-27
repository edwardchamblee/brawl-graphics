"""Decode Odin skeletal animations into per-node keyframe tracks.

Three on-disk layouts exist and are selected from the ``animation``
descriptor found in the ``SC_odin_format`` root extension:

* **raw** - ``nodes`` + ``accessor``: float TRS (3+4+3) per node per frame.
* **packed** - ``packed`` without ``uintAccessor``: int16 deltas over a base
  transform, node base data is 12 floats (T, R, S, tMul, sMul).
* **RLE packed** - ``packed`` with ``uintAccessor``: like packed, but the base
  rotation lives in ``uintAccessor`` (8 floats of node data) and every node's
  frame stream is run-length encoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..gltf.accessors import read_accessor

FLAG_TRACKTIME = 1 << 0
FLAG_ROTATION = 1 << 1
FLAG_TRANSLATION = 1 << 2
FLAG_SCALE = 1 << 3
FLAG_SCALE3D = 1 << 4

QUAT_SCALE = 32767.0


class AnimationError(ValueError):
    pass


@dataclass
class NodeTrack:
    node: int
    frame_count: int
    translation: np.ndarray  # (F, 3) float32
    rotation: np.ndarray  # (F, 4) float32, xyzw
    scale: np.ndarray  # (F, 3) float32
    animated: dict[str, bool] = field(default_factory=dict)

    def channel(self, path: str) -> np.ndarray:
        return {"translation": self.translation, "rotation": self.rotation, "scale": self.scale}[
            path
        ]


@dataclass
class OdinAnimation:
    frame_rate: float
    first_frame: float
    last_frame: float
    keyframe_count: int
    tracks: list[NodeTrack]

    @property
    def frame_count(self) -> int:
        return max([t.frame_count for t in self.tracks] + [self.keyframe_count])

    @property
    def duration(self) -> float:
        return max(self.frame_count - 1, 0) / self.frame_rate


def _flag(flags: int, bit: int) -> bool:
    return bool(flags & bit)


def _elements_per_frame(flags: int) -> int:
    count = 0
    if _flag(flags, FLAG_TRACKTIME):
        count += 1
    if _flag(flags, FLAG_ROTATION):
        count += 4
    if _flag(flags, FLAG_TRANSLATION):
        count += 3
    if _flag(flags, FLAG_SCALE):
        count += 3 if _flag(flags, FLAG_SCALE3D) else 1
    return count


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(quats, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (quats / lengths).astype(np.float32)


class _Cursor:
    """Sequential reader over the normalized (int16) data stream."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.position = 0
        self.consumed = 0

    def read(self) -> int:
        if self.position >= len(self.data):
            raise AnimationError("animation data stream ended prematurely")
        value = int(self.data[self.position])
        self.position += 1
        self.consumed += 1
        return value

    def read_frame(self, flags: int, out: dict[str, np.ndarray], frame: int) -> None:
        if _flag(flags, FLAG_TRACKTIME):
            self.read()
        if _flag(flags, FLAG_ROTATION):
            out["rotation"][frame] = [self.read() for _ in range(4)]
        if _flag(flags, FLAG_TRANSLATION):
            out["translation"][frame] = [self.read() for _ in range(3)]
        if _flag(flags, FLAG_SCALE):
            if _flag(flags, FLAG_SCALE3D):
                out["scale"][frame] = [self.read() for _ in range(3)]
            else:
                out["scale"][frame] = self.read()


def _empty_normalized(frame_count: int) -> dict[str, np.ndarray]:
    return {
        "rotation": np.zeros((frame_count, 4), dtype=np.int32),
        "translation": np.zeros((frame_count, 3), dtype=np.int32),
        "scale": np.zeros((frame_count, 3), dtype=np.int32),
    }


def _read_rle_frames(cursor: _Cursor, flags: int, frame_count: int, data_size: int) -> dict:
    out = _empty_normalized(frame_count)
    cursor.consumed = 0
    if flags == 0 or frame_count == 0:
        return out

    frame = 0
    while frame < frame_count:
        run = cursor.read()
        if run == 0:
            raise AnimationError("zero run length in RLE animation")
        if abs(run) > frame_count - frame:
            raise AnimationError("RLE run exceeds remaining frames")
        if run > 0:
            for _ in range(run):
                cursor.read_frame(flags, out, frame)
                frame += 1
        else:
            for _ in range(-run):
                if frame > 0:
                    for key in out:
                        out[key][frame] = out[key][frame - 1]
                frame += 1

    if cursor.consumed != data_size:
        raise AnimationError(
            f"RLE node consumed {cursor.consumed} values but descriptor says {data_size}"
        )
    return out


def _read_sequential_frames(cursor: _Cursor, flags: int, frame_count: int) -> dict:
    out = _empty_normalized(frame_count)
    for frame in range(frame_count):
        cursor.read_frame(flags, out, frame)
    return out


def _read_frame_major(
    data: np.ndarray, flags: int, frame_count: int, frame_stride: int, node_offset: int
) -> dict:
    out = _empty_normalized(frame_count)
    for frame in range(frame_count):
        cursor = _Cursor(data)
        cursor.position = frame * frame_stride + node_offset
        cursor.read_frame(flags, out, frame)
    return out


def _denormalize(
    flags: int,
    frame_count: int,
    normalized: dict[str, np.ndarray],
    base_t: np.ndarray,
    base_r: np.ndarray,
    base_s: np.ndarray,
    t_mul: float,
    s_mul: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = np.tile(base_t.astype(np.float32), (frame_count, 1))
    rotation = np.tile(base_r.astype(np.float32), (frame_count, 1))
    scale = np.tile(base_s.astype(np.float32), (frame_count, 1))

    if _flag(flags, FLAG_TRANSLATION):
        translation = translation + normalized["translation"].astype(np.float32) * t_mul
    if _flag(flags, FLAG_ROTATION):
        rotation = normalized["rotation"].astype(np.float32) / QUAT_SCALE
    if _flag(flags, FLAG_SCALE):
        scale = scale + normalized["scale"].astype(np.float32) * s_mul

    return translation, _normalize_quaternions(rotation), scale


def _read_packed(gltf: dict, binary: bytes, descriptor: dict) -> OdinAnimation:
    packed = descriptor["packed"]
    nodes: list[dict] = packed.get("nodes") or []
    data = read_accessor(gltf, binary, packed["dataAccessor"]).reshape(-1)
    node_base = read_accessor(gltf, binary, packed["nodeAccessor"]).reshape(-1).astype(np.float64)

    rotations = None
    if packed.get("uintAccessor") is not None:
        rotations = read_accessor(gltf, binary, packed["uintAccessor"]).reshape(-1)
        base_stride = 8
    else:
        base_stride = 12

    flags_list = [int(n.get("flags", 0)) for n in nodes]
    tracktime = [_flag(f, FLAG_TRACKTIME) for f in flags_list]
    if any(tracktime) and not all(tracktime):
        raise AnimationError("mixed tracktime usage between nodes is not supported")
    frame_major = rotations is None and not any(tracktime)
    frame_stride = sum(_elements_per_frame(f) for f in flags_list)

    cursor = _Cursor(data)
    node_offset = 0
    tracks: list[NodeTrack] = []

    for index, node in enumerate(nodes):
        flags = flags_list[index]
        frame_count = int(node.get("frameCount", 0))
        base = node_base[index * base_stride : (index + 1) * base_stride]
        if len(base) < base_stride:
            raise AnimationError("node base data is truncated")

        if rotations is not None:
            base_t, base_s = base[0:3], base[3:6]
            t_mul, s_mul = float(base[6]), float(base[7])
            base_r = rotations[index * 4 : index * 4 + 4].astype(np.float64) / QUAT_SCALE
        else:
            base_t, base_r, base_s = base[0:3], base[3:7], base[7:10]
            t_mul, s_mul = float(base[10]), float(base[11])

        if rotations is not None:
            normalized = _read_rle_frames(cursor, flags, frame_count, int(node.get("dataSize", 0)))
        elif frame_major:
            normalized = _read_frame_major(data, flags, frame_count, frame_stride, node_offset)
            node_offset += _elements_per_frame(flags)
        else:
            normalized = _read_sequential_frames(cursor, flags, frame_count)

        translation, rotation, scale = _denormalize(
            flags, frame_count, normalized, base_t, base_r, base_s, t_mul, s_mul
        )
        tracks.append(
            NodeTrack(
                node=int(node.get("nodeIndex", 0)),
                frame_count=frame_count,
                translation=translation,
                rotation=rotation,
                scale=scale,
                animated={
                    "translation": _flag(flags, FLAG_TRANSLATION),
                    "rotation": _flag(flags, FLAG_ROTATION),
                    "scale": _flag(flags, FLAG_SCALE),
                },
            )
        )

    return OdinAnimation(
        frame_rate=float(descriptor.get("frameRate") or 30.0),
        first_frame=float(descriptor.get("firstFrame") or 0.0),
        last_frame=float(descriptor.get("lastFrame") or 0.0),
        keyframe_count=int(descriptor.get("keyframeCount") or descriptor.get("keyframesCount") or 1),
        tracks=tracks,
    )


def _read_raw(gltf: dict, binary: bytes, descriptor: dict) -> OdinAnimation:
    node_indices: list[int] = list(descriptor["nodes"])
    keyframe_count = int(descriptor.get("keyframesCount") or descriptor.get("keyframeCount") or 1)
    buffer = read_accessor(gltf, binary, descriptor["accessor"]).reshape(-1).astype(np.float32)

    per_node_frames: list[int]
    keyframe_counts = descriptor.get("keyframeCounts")
    if keyframe_counts:
        nodes_per_keyframe = descriptor.get("nodesNumberPerKeyframe") or []
        per_node_frames = [
            int(count)
            for i, count in enumerate(keyframe_counts)
            for _ in range(int(nodes_per_keyframe[i]))
        ]
    else:
        per_node_frames = [keyframe_count] * len(node_indices)

    if len(per_node_frames) != len(node_indices):
        raise AnimationError("keyframe mapping does not match node count")

    frames = buffer.reshape(-1, 10)
    if len(frames) != sum(per_node_frames):
        raise AnimationError("raw animation buffer size does not match keyframe counts")

    tracks: list[NodeTrack] = []
    offset = 0
    for node, count in zip(node_indices, per_node_frames):
        chunk = frames[offset : offset + count]
        offset += count
        tracks.append(
            NodeTrack(
                node=int(node),
                frame_count=count,
                translation=np.ascontiguousarray(chunk[:, 0:3]),
                rotation=_normalize_quaternions(chunk[:, 3:7]),
                scale=np.ascontiguousarray(chunk[:, 7:10]),
                animated={"translation": True, "rotation": True, "scale": True},
            )
        )

    return OdinAnimation(
        frame_rate=float(descriptor.get("frameRate") or 30.0),
        first_frame=float(descriptor.get("firstFrame") or 0.0),
        last_frame=float(descriptor.get("lastFrame") or 0.0),
        keyframe_count=keyframe_count,
        tracks=tracks,
    )


def read_animation(gltf: dict[str, Any], binary: bytes, descriptor: dict[str, Any]) -> OdinAnimation:
    """Decode the ``animation`` descriptor of the ``SC_odin_format`` extension."""
    if descriptor.get("packed") is not None:
        return _read_packed(gltf, binary, descriptor)
    if descriptor.get("nodes") is not None and descriptor.get("accessor") is not None:
        return _read_raw(gltf, binary, descriptor)
    raise AnimationError("unknown Odin animation layout")


def find_animation_descriptor(gltf: dict[str, Any]) -> dict[str, Any] | None:
    odin = (gltf.get("extensions") or {}).get("SC_odin_format") or {}
    return odin.get("animation")
