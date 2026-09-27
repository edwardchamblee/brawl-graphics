"""Odin vertex attribute names and storage formats.

Odin meshes store vertex data as interleaved GPU streams. Each attribute is
identified by the shader attribute name (``a_pos``, ``a_uv0``...) and a
numeric storage format.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# Shader attribute name -> glTF attribute semantic
ATTRIBUTE_SEMANTICS: dict[str, str] = {
    "a_pos": "POSITION",
    "a_normal": "NORMAL",
    "a_uv0": "TEXCOORD_0",
    "a_uv1": "TEXCOORD_1",
    "a_color": "COLOR_0",
    "a_color1": "COLOR_1",
    "a_boneindex": "JOINTS_0",
    "a_boneweights": "WEIGHTS_0",
    "a_tangent": "TANGENT",
}


class AttributeFormat(IntEnum):
    UByte2 = 1
    UByte3 = 2
    UByte4 = 3
    Byte2 = 4
    Byte3 = 5
    Byte4 = 6
    UByte2Norm = 7
    UByte3Norm = 8
    UByte4Norm = 9
    Byte2Norm = 10
    Byte3Norm = 11
    Byte4Norm = 12
    UShort2 = 13
    UShort3 = 14
    UShort4 = 15
    Short2 = 16
    Short3 = 17
    Short4 = 18
    UShort2Norm = 19
    UShort3Norm = 20
    UShort4Norm = 21
    Short2Norm = 22
    Short3Norm = 23
    Short4Norm = 24
    Half2 = 25
    Half3 = 26
    Half4 = 27
    Float = 28
    Float2 = 29
    Float3 = 30
    Float4 = 31
    Int = 32
    Int2 = 33
    Int3 = 34
    Int4 = 35
    UInt = 36
    UInt2 = 37
    UInt3 = 38
    UInt4 = 39
    Int1010102Norm = 40
    UInt1010102Norm = 41
    Float2x2 = 100
    Float3x3 = 101
    Float4x4 = 102


@dataclass(frozen=True)
class FormatInfo:
    dtype: np.dtype
    components: int
    normalized: bool

    @property
    def byte_size(self) -> int:
        return self.dtype.itemsize * self.components


def _info(dtype: type, components: int, normalized: bool = False) -> FormatInfo:
    return FormatInfo(np.dtype(dtype), components, normalized)


FORMAT_INFO: dict[AttributeFormat, FormatInfo] = {
    AttributeFormat.UByte2: _info(np.uint8, 2),
    AttributeFormat.UByte3: _info(np.uint8, 3),
    AttributeFormat.UByte4: _info(np.uint8, 4),
    AttributeFormat.Byte2: _info(np.int8, 2),
    AttributeFormat.Byte3: _info(np.int8, 3),
    AttributeFormat.Byte4: _info(np.int8, 4),
    AttributeFormat.UByte2Norm: _info(np.uint8, 2, True),
    AttributeFormat.UByte3Norm: _info(np.uint8, 3, True),
    AttributeFormat.UByte4Norm: _info(np.uint8, 4, True),
    AttributeFormat.Byte2Norm: _info(np.int8, 2, True),
    AttributeFormat.Byte3Norm: _info(np.int8, 3, True),
    AttributeFormat.Byte4Norm: _info(np.int8, 4, True),
    AttributeFormat.UShort2: _info(np.uint16, 2),
    AttributeFormat.UShort3: _info(np.uint16, 3),
    AttributeFormat.UShort4: _info(np.uint16, 4),
    AttributeFormat.Short2: _info(np.int16, 2),
    AttributeFormat.Short3: _info(np.int16, 3),
    AttributeFormat.Short4: _info(np.int16, 4),
    AttributeFormat.UShort2Norm: _info(np.uint16, 2, True),
    AttributeFormat.UShort3Norm: _info(np.uint16, 3, True),
    AttributeFormat.UShort4Norm: _info(np.uint16, 4, True),
    AttributeFormat.Short2Norm: _info(np.int16, 2, True),
    AttributeFormat.Short3Norm: _info(np.int16, 3, True),
    AttributeFormat.Short4Norm: _info(np.int16, 4, True),
    AttributeFormat.Half2: _info(np.float16, 2),
    AttributeFormat.Half3: _info(np.float16, 3),
    AttributeFormat.Half4: _info(np.float16, 4),
    AttributeFormat.Float: _info(np.float32, 1),
    AttributeFormat.Float2: _info(np.float32, 2),
    AttributeFormat.Float3: _info(np.float32, 3),
    AttributeFormat.Float4: _info(np.float32, 4),
    AttributeFormat.Int: _info(np.int32, 1),
    AttributeFormat.Int2: _info(np.int32, 2),
    AttributeFormat.Int3: _info(np.int32, 3),
    AttributeFormat.Int4: _info(np.int32, 4),
    AttributeFormat.UInt: _info(np.uint32, 1),
    AttributeFormat.UInt2: _info(np.uint32, 2),
    AttributeFormat.UInt3: _info(np.uint32, 3),
    AttributeFormat.UInt4: _info(np.uint32, 4),
    AttributeFormat.Int1010102Norm: _info(np.uint32, 1, True),
    AttributeFormat.UInt1010102Norm: _info(np.uint32, 1, True),
    AttributeFormat.Float2x2: _info(np.float32, 4),
    AttributeFormat.Float3x3: _info(np.float32, 9),
    AttributeFormat.Float4x4: _info(np.float32, 16),
}


def format_info(value: int) -> FormatInfo:
    try:
        return FORMAT_INFO[AttributeFormat(value)]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"unsupported Odin attribute format {value}") from exc
