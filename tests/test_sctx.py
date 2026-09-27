from __future__ import annotations

import io
import struct

import flatbuffers
import numpy as np
import pytest
import zstandard
from PIL import Image

from bsx.textures.resolver import TextureResolver, split_texture_reference
from bsx.textures.sctx import SctxError, decode_rgba, parse_sctx, sctx_to_png


def _header(
    pixel_type: int, width: int, height: int, levels: int, flags: int, length: int
) -> bytes:
    builder = flatbuffers.Builder(64)
    builder.StartObject(11)
    builder.PrependUint32Slot(1, pixel_type, 0)
    builder.PrependUint16Slot(2, width, 0)
    builder.PrependUint16Slot(3, height, 0)
    builder.PrependUint8Slot(4, levels, 1)
    builder.PrependUint32Slot(6, flags, 0)
    builder.PrependInt32Slot(7, length, 0)
    root = builder.EndObject()
    builder.Finish(root, file_identifier=b"SCTX")
    return bytes(builder.Output())


def _mip(width: int, height: int, offset: int) -> bytes:
    builder = flatbuffers.Builder(32)
    builder.StartObject(4)
    builder.PrependUint16Slot(0, width, 0)
    builder.PrependUint16Slot(1, height, 0)
    builder.PrependUint32Slot(2, offset, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return bytes(builder.Output())


def make_sctx(
    pixels: bytes, width: int, height: int, pixel_type: int, compress: bool, padding: bool
) -> bytes:
    flags = (1 if compress else 0) | (8 if padding else 0)
    header = _header(pixel_type, width, height, 1, flags, len(pixels))
    mip = _mip(width, height, 0)
    out = bytearray()
    out += struct.pack("<I", len(header)) + header
    mips = struct.pack("<I", len(mip)) + mip
    out += struct.pack("<I", len(mips)) + mips
    if padding:
        out += b"\x00" * (-len(out) % 16)
    out += zstandard.ZstdCompressor().compress(pixels) if compress else pixels
    return bytes(out)


@pytest.mark.parametrize("compress,padding", [(False, False), (True, False), (True, True)])
def test_parse_and_decode_raw_rgba(compress, padding):
    rgba = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    data = make_sctx(rgba.tobytes(), 4, 4, 70, compress, padding)

    texture = parse_sctx(data)
    assert (texture.width, texture.height) == (4, 4)
    assert texture.format_name == "RGBA8"
    assert not texture.is_astc
    assert texture.levels[0].width == 4

    image = decode_rgba(texture)
    np.testing.assert_array_equal(np.asarray(image), rgba)

    png = sctx_to_png(data)
    assert Image.open(io.BytesIO(png)).size == (4, 4)


def test_astc_format_table():
    from bsx.textures.sctx import ASTC_FORMATS

    assert ASTC_FORMATS[186] == (4, 4)
    assert ASTC_FORMATS[194] == (8, 8)
    assert ASTC_FORMATS[212] == (8, 8)
    assert ASTC_FORMATS[218] == (12, 12)
    assert (
        parse_sctx(make_sctx(b"\x00" * 16, 4, 4, 186, False, False)).format_name
        == "ASTC 4x4 (sRGB)"
    )


def test_bad_identifier():
    with pytest.raises(SctxError):
        parse_sctx(struct.pack("<I", 8) + b"\x00\x00\x00\x00NOPE" + b"\x00" * 8)


def test_texture_resolver_prefers_local_files(tmp_path):
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    (tmp_path / "hero_tex.sctx").write_bytes(make_sctx(rgba.tobytes(), 2, 2, 70, True, False))
    calls = []

    def fetcher(remote: str):
        calls.append(remote)
        return None

    resolver = TextureResolver(lookup_dirs=[tmp_path], fetcher=fetcher)
    resolved = resolver.resolve("hero_tex.png#repeat")
    assert resolved is not None
    assert resolved.repeat
    assert resolved.source.endswith("hero_tex.sctx")
    assert Image.open(io.BytesIO(resolved.png)).size == (2, 2)
    assert calls == []

    assert resolver.resolve("missing.png") is None
    assert calls[0] == "sc3d/missing.png"
    assert split_texture_reference("a.png#repeat+clamp") == ("a.png", ["repeat", "clamp"])
