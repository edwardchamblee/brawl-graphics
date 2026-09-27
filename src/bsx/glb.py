"""GLB container reading and writing.

Supercell's Odin files are regular GLB containers whose first chunk is a
FlatBuffer (``FLA2``) instead of ``JSON``. Everything else (header, ``BIN``
chunk) follows the glTF 2.0 specification.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

GLB_MAGIC = b"glTF"
CHUNK_JSON = b"JSON"
CHUNK_BIN = b"BIN\x00"
CHUNK_FLAT = b"FLA2"


class GlbError(ValueError):
    pass


@dataclass
class GlbDocument:
    gltf: dict[str, Any]
    binary: bytes
    kind: str  # "JSON" or "FLA2"

    @property
    def is_odin(self) -> bool:
        required = self.gltf.get("extensionsRequired") or []
        used = self.gltf.get("extensionsUsed") or []
        extensions = self.gltf.get("extensions") or {}
        return (
            self.kind == "FLA2"
            or "SC_odin_format" in required
            or "SC_odin_format" in used
            or "SC_odin_format" in extensions
        )


def read_glb(data: bytes) -> GlbDocument:
    if data[:4] != GLB_MAGIC:
        raise GlbError("not a GLB file (bad magic)")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise GlbError(f"unsupported GLB version {version}")
    if length != len(data):
        raise GlbError(f"GLB length mismatch (header {length}, actual {len(data)})")

    offset = 12
    chunks: list[tuple[bytes, bytes]] = []
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        start = offset + 8
        end = start + chunk_length
        if end > len(data):
            raise GlbError("truncated GLB chunk")
        chunks.append((chunk_type, data[start:end]))
        offset = end

    if not chunks:
        raise GlbError("GLB has no chunks")

    first_type, first_data = chunks[0]
    if first_type == CHUNK_FLAT:
        from .flat import deserialize_root

        gltf = deserialize_root(first_data)
        kind = "FLA2"
    elif first_type == CHUNK_JSON:
        gltf = json.loads(first_data.decode("utf-8"))
        kind = "JSON"
    else:
        raise GlbError(f"first chunk must be JSON or FLA2, got {first_type!r}")

    binary = b""
    for chunk_type, chunk_data in chunks[1:]:
        if chunk_type == CHUNK_BIN:
            binary = chunk_data
            break

    return GlbDocument(gltf=gltf, binary=binary, kind=kind)


def read_glb_file(path: str) -> GlbDocument:
    with open(path, "rb") as f:
        return read_glb(f.read())


def _pad(data: bytes, pad_byte: bytes) -> bytes:
    remainder = len(data) % 4
    if remainder:
        data += pad_byte * (4 - remainder)
    return data


def write_glb(gltf: dict[str, Any], binary: bytes) -> bytes:
    json_bytes = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = _pad(bytes(binary), b"\x00")

    total = 12 + 8 + len(json_bytes) + (8 + len(bin_bytes) if bin_bytes else 0)
    out = bytearray()
    out += struct.pack("<4sII", GLB_MAGIC, 2, total)
    out += struct.pack("<I4s", len(json_bytes), CHUNK_JSON) + json_bytes
    if bin_bytes:
        out += struct.pack("<I4s", len(bin_bytes), CHUNK_BIN) + bin_bytes
    return bytes(out)
