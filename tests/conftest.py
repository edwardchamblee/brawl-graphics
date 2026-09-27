from __future__ import annotations

import struct

import flatbuffers
import numpy as np
import pytest
from flatbuffers import flexbuffers

from bsx.flat import gltf_generated as fb
from bsx.glb import GlbDocument


def make_glb_bytes(first_chunk_type: bytes, first_chunk: bytes, binary: bytes) -> bytes:
    def pad(data: bytes, byte: bytes) -> bytes:
        return data + byte * (-len(data) % 4)

    first = pad(first_chunk, b"\x00" if first_chunk_type == b"FLA2" else b" ")
    binary = pad(binary, b"\x00")
    body = struct.pack("<I4s", len(first), first_chunk_type) + first
    if binary:
        body += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def build_flat_root(builder_fn) -> bytes:
    """Run ``builder_fn(builder)`` returning a Root offset and finish the buffer."""
    builder = flatbuffers.Builder(1024)
    root = builder_fn(builder)
    builder.Finish(root)
    return bytes(builder.Output())


def flex(builder: flatbuffers.Builder, value) -> int:
    return builder.CreateByteVector(flexbuffers.Dumps(value))


@pytest.fixture
def synthetic_odin_document() -> GlbDocument:
    """A tiny hand-made Odin model: 3 bones, one skinned quad, two primitives."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [9, 9, 9]], dtype=np.float32)
    joints = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    # packed weights: x in bits 21.., y bits 10..20, z bits 0..9 ; w0 = 1 - x - y - z
    packed = np.array([2047 << 21, 0, 1024 << 21, 512 << 10, 0], dtype=np.uint32)
    normals = np.array([[0, 0, 127, 0]] * 5, dtype=np.int8)
    uvs = (np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]) * 32767).astype(np.int16)

    stream0 = bytearray()
    for i in range(5):
        stream0 += positions[i].tobytes() + joints[i].tobytes() + packed[i : i + 1].tobytes()
    stream1 = bytearray()
    for i in range(5):
        stream1 += normals[i].tobytes() + uvs[i].tobytes()

    indices_a = np.array([0, 1, 2], dtype=np.uint16)
    indices_b = np.array([0, 2, 3], dtype=np.uint16)
    ibm = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (3, 1))
    ibm[1, 12] = -1.0  # joint 1 sits at x=1

    binary = bytearray()
    idx_offset = len(binary)
    binary += indices_a.tobytes() + indices_b.tobytes()
    binary += b"\x00" * (-len(binary) % 4)
    ibm_offset = len(binary)
    binary += ibm.tobytes()
    vertex_offset = len(binary)
    binary += stream0 + stream1

    gltf = {
        "asset": {"version": "2.0"},
        "extensionsRequired": ["SC_odin_format"],
        "extensionsUsed": ["SC_odin_format"],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": 12},
            {"buffer": 0, "byteOffset": ibm_offset, "byteLength": ibm.nbytes},
            {"buffer": 0, "byteOffset": vertex_offset, "byteLength": len(stream0) + len(stream1)},
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 0, "byteOffset": 6, "componentType": 5123, "count": 3, "type": "SCALAR"},
            # Supercell flag bits in the upper half word
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 0x10000 | 5126,
                "count": 3,
                "type": "MAT4",
            },
        ],
        "nodes": [
            {"name": "CHARACTER", "scale": [0, 0, 0]},
            {"name": "root_bone", "extensions": {"SC_odin_format": {"parent": 0}}},
            {
                "name": "child_bone",
                "translation": [1, 0, 0],
                "extensions": {"SC_odin_format": {"parent": 1}},
            },
            {
                "name": "tip_bone",
                "translation": [0, 1, 0],
                "extensions": {"SC_odin_format": {"parent": 2}},
            },
            {"name": "quadGeo", "mesh": 0, "extensions": {"SC_odin_format": {"parent": 0}}},
        ],
        "skins": [{"joints": [1, 2, 3], "inverseBindMatrices": 2}],
        "meshes": [
            {
                "primitives": [
                    {
                        "indices": 0,
                        "material": 0,
                        "extensions": {"SC_odin_format": {"meshDataInfoIndex": 0}},
                    },
                    {
                        "indices": 1,
                        "material": 1,
                        "extensions": {"SC_odin_format": {"meshDataInfoIndex": 0}},
                    },
                ],
                "extensions": {"SC_odin_format": {"bounds": [[0, 0, 0], [1, 1, 0]]}},
            }
        ],
        "extensions": {
            "SC_odin_format": {
                "bufferView": 2,
                "materials": [
                    {
                        "name": "body_mat",
                        "shader": "uber",
                        "blendMode": 4,
                        "constants": ["DIFFUSE"],
                        "variables": {
                            "floats": {"cutout": 0.0},
                            "textures": {"diffuseTex": "body.png"},
                        },
                    },
                    {
                        "name": "glass_mat",
                        "shader": "uber",
                        "blendMode": 5,
                        "variables": {"floatVectors": {"diffuse": [0.2, 0.4, 0.6, 0.5]}},
                    },
                ],
                "meshDataInfos": [
                    {
                        "vertexDescriptors": [
                            {
                                "offset": 0,
                                "stride": 20,
                                "attributes": [
                                    {"name": "a_pos", "format": 30, "offset": 0},
                                    {"name": "a_boneindex", "format": 3, "offset": 12},
                                    {"name": "a_boneweights", "format": 36, "offset": 16},
                                ],
                            },
                            {
                                "offset": len(stream0),
                                "stride": 8,
                                "attributes": [
                                    {"name": "a_normal", "format": 12, "offset": 0},
                                    {"name": "a_uv0", "format": 22, "offset": 4},
                                ],
                            },
                        ]
                    }
                ],
            }
        },
    }
    return GlbDocument(gltf=gltf, binary=bytes(binary), kind="FLA2")


def make_rle_animation_document(
    nodes: list[dict], node_specs: list[dict], frame_count: int
) -> GlbDocument:
    """Build an RLE-packed Odin animation.

    ``node_specs`` items: ``{"node": idx, "flags": f, "base": {"t": [...], "r": [...], "s": [...],
    "tmul": x, "smul": y}, "stream": [int16 values including run lengths]}``.
    """
    data = []
    node_base = []
    rotations = []
    packed_nodes = []
    for spec in node_specs:
        base = spec["base"]
        node_base += list(base["t"]) + list(base["s"]) + [base["tmul"], base["smul"]]
        rotations += [int(round(v * 32767)) for v in base["r"]]
        data += spec["stream"]
        packed_nodes.append(
            {
                "nodeIndex": spec["node"],
                "flags": spec["flags"],
                "frameCount": frame_count,
                "dataSize": len(spec["stream"]),
            }
        )

    data_arr = np.array(data, dtype=np.int16)
    base_arr = np.array(node_base, dtype=np.float32)
    rot_arr = np.array(rotations, dtype=np.int16)

    binary = bytearray()
    offsets = []
    for arr in (data_arr, base_arr, rot_arr):
        binary += b"\x00" * (-len(binary) % 4)
        offsets.append(len(binary))
        binary += arr.tobytes()

    gltf = {
        "asset": {"version": "2.0"},
        "extensionsRequired": ["SC_odin_format"],
        "extensionsUsed": ["SC_odin_format"],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": data_arr.nbytes},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": base_arr.nbytes},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": rot_arr.nbytes},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 0x10000 | 5122,
                "count": len(data_arr),
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "componentType": 0x20000 | 5126,
                "count": len(base_arr),
                "type": "SCALAR",
            },
            {
                "bufferView": 2,
                "componentType": 0x30000 | 5122,
                "count": len(rot_arr),
                "type": "SCALAR",
            },
        ],
        "nodes": nodes,
        "extensions": {
            "SC_odin_format": {
                "animation": {
                    "firstFrame": 0.0,
                    "lastFrame": float(frame_count - 1),
                    "frameRate": 30.0,
                    "keyframeCount": frame_count,
                    "packed": {
                        "dataAccessor": 0,
                        "nodeAccessor": 1,
                        "uintAccessor": 2,
                        "stride": 0,
                        "nodes": packed_nodes,
                    },
                }
            }
        },
    }
    return GlbDocument(gltf=gltf, binary=bytes(binary), kind="FLA2")


__all__ = ["fb", "make_glb_bytes", "build_flat_root", "flex", "make_rle_animation_document"]
