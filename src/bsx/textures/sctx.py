"""Parser and decoder for Supercell ``.sctx`` textures.

Layout (little endian)::

    u32 header_length
    FlatBuffer  Header        (file identifier "SCTX")
    u32 mipmaps_length
    repeat levels_count:
        u32 mip_length
        FlatBuffer MipMap     (width, height, offset, hash)
    [padding to 16 bytes when the use_padding flag is set]
    texture data: zstd frame or raw bytes of texture_length

The texture data holds every mip level back to back at the offsets given by
the MipMap tables. Pixel data is either raw RGBA8 or an ASTC block format.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import flatbuffers
from flatbuffers import number_types as N
from flatbuffers.table import Table

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
SCTX_IDENTIFIER = b"SCTX"

FLAG_USE_COMPRESSION = 1 << 0
FLAG_USE_PADDING = 1 << 3

# ScPixel::Type values that this module knows how to decode.
RAW_FORMATS: dict[int, str] = {
    70: "RGBA8",  # RGBA8Unorm
    71: "RGBA8",  # RGBA8Unorm_sRGB
    80: "BGRA8",  # BGRA8Unorm
    81: "BGRA8",  # BGRA8Unorm_sRGB
    266: "RGB8",  # RGB8Unorm
    264: "L8",  # Luminance
    1: "A8",  # A8Unorm
}

_ASTC_BLOCKS = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
]

ASTC_FORMATS: dict[int, tuple[int, int]] = {}
# sRGB: 186..190 (4x4..6x6) then 192..200 (8x5..12x12)
for _i, _block in enumerate(_ASTC_BLOCKS[:5]):
    ASTC_FORMATS[186 + _i] = _block
for _i, _block in enumerate(_ASTC_BLOCKS[5:]):
    ASTC_FORMATS[192 + _i] = _block
# Linear: 204..208 then 210..218
for _i, _block in enumerate(_ASTC_BLOCKS[:5]):
    ASTC_FORMATS[204 + _i] = _block
for _i, _block in enumerate(_ASTC_BLOCKS[5:]):
    ASTC_FORMATS[210 + _i] = _block


class SctxError(ValueError):
    pass


@dataclass
class MipLevel:
    width: int
    height: int
    offset: int


@dataclass
class SctxTexture:
    pixel_type: int
    width: int
    height: int
    flags: int
    texture_length: int
    levels: list[MipLevel] = field(default_factory=list)
    data: bytes = b""

    @property
    def is_astc(self) -> bool:
        return self.pixel_type in ASTC_FORMATS

    @property
    def astc_block(self) -> tuple[int, int] | None:
        return ASTC_FORMATS.get(self.pixel_type)

    @property
    def format_name(self) -> str:
        block = self.astc_block
        if block:
            colorspace = "sRGB" if self.pixel_type < 204 else "linear"
            return f"ASTC {block[0]}x{block[1]} ({colorspace})"
        return RAW_FORMATS.get(self.pixel_type, f"pixel type {self.pixel_type}")

    def level_bytes(self, level: int = 0) -> bytes:
        info = self.levels[level]
        end = self.levels[level + 1].offset if level + 1 < len(self.levels) else len(self.data)
        return self.data[info.offset : end]


def _table(buffer: bytes) -> Table:
    root = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buffer, 0)
    return Table(buffer, root)


def _scalar(table: Table, slot: int, flags, default):
    offset = table.Offset(4 + 2 * slot)
    return table.Get(flags, offset + table.Pos) if offset else default


def parse_sctx(data: bytes) -> SctxTexture:
    if len(data) < 12:
        raise SctxError("file too small")
    (header_length,) = struct.unpack_from("<I", data, 0)
    header = data[4 : 4 + header_length]
    if header[4:8] != SCTX_IDENTIFIER:
        raise SctxError("missing SCTX identifier")

    table = _table(header)
    texture = SctxTexture(
        pixel_type=_scalar(table, 1, N.Uint32Flags, 0),
        width=_scalar(table, 2, N.Uint16Flags, 0),
        height=_scalar(table, 3, N.Uint16Flags, 0),
        flags=_scalar(table, 6, N.Uint32Flags, 0),
        texture_length=_scalar(table, 7, N.Int32Flags, 0),
    )
    levels_count = _scalar(table, 4, N.Uint8Flags, 1)

    position = 4 + header_length
    (mipmaps_length,) = struct.unpack_from("<I", data, position)
    position += 4
    data_start = position + mipmaps_length

    for _ in range(levels_count):
        (mip_length,) = struct.unpack_from("<I", data, position)
        mip = _table(data[position + 4 : position + 4 + mip_length])
        texture.levels.append(
            MipLevel(
                width=_scalar(mip, 0, N.Uint16Flags, 0),
                height=_scalar(mip, 1, N.Uint16Flags, 0),
                offset=_scalar(mip, 2, N.Uint32Flags, 0),
            )
        )
        position += 4 + mip_length

    if not texture.levels:
        texture.levels.append(MipLevel(texture.width, texture.height, 0))

    if texture.flags & FLAG_USE_PADDING:
        data_start = (data_start + 15) & ~15

    payload = data[data_start:]
    if payload[:4] == ZSTD_MAGIC:
        import zstandard

        texture.data = zstandard.ZstdDecompressor().decompressobj().decompress(payload)
    else:
        texture.data = bytes(payload[: texture.texture_length])

    if len(texture.data) < texture.texture_length:
        raise SctxError(
            f"texture data truncated ({len(texture.data)} of {texture.texture_length} bytes)"
        )
    return texture


def decode_rgba(texture: SctxTexture, level: int = 0):
    """Return a ``PIL.Image`` in RGBA for one mip level."""
    import numpy as np
    from PIL import Image

    info = texture.levels[level]
    width, height = info.width, info.height
    payload = texture.level_bytes(level)

    block = texture.astc_block
    if block is not None:
        import texture2ddecoder

        bw, bh = block
        blocks = ((width + bw - 1) // bw) * ((height + bh - 1) // bh)
        needed = blocks * 16
        if len(payload) < needed:
            raise SctxError("ASTC payload too small for the declared dimensions")
        bgra = texture2ddecoder.decode_astc(payload[:needed], width, height, bw, bh)
        array = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)[..., [2, 1, 0, 3]]
        return Image.fromarray(np.ascontiguousarray(array), "RGBA")

    raw = RAW_FORMATS.get(texture.pixel_type)
    if raw is None:
        raise SctxError(f"unsupported pixel type {texture.pixel_type}")
    if raw == "RGBA8":
        return Image.frombytes("RGBA", (width, height), payload[: width * height * 4])
    if raw == "BGRA8":
        array = np.frombuffer(payload[: width * height * 4], dtype=np.uint8)
        array = array.reshape(height, width, 4)[..., [2, 1, 0, 3]]
        return Image.fromarray(np.ascontiguousarray(array), "RGBA")
    if raw == "RGB8":
        return Image.frombytes("RGB", (width, height), payload[: width * height * 3]).convert("RGBA")
    if raw == "L8":
        return Image.frombytes("L", (width, height), payload[: width * height]).convert("RGBA")
    if raw == "A8":
        alpha = Image.frombytes("L", (width, height), payload[: width * height])
        image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        image.putalpha(alpha)
        return image
    raise SctxError(f"unsupported pixel type {texture.pixel_type}")  # pragma: no cover


def sctx_to_png(data: bytes, level: int = 0) -> bytes:
    """Decode an ``.sctx`` file into PNG bytes."""
    import io

    image = decode_rgba(parse_sctx(data), level)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
