"""Decoder for the ``FLA2`` chunk of Supercell Odin ``.glb`` files.

Supercell stores the glTF document as a FlatBuffer whose schema mirrors the
glTF 2.0 JSON schema one-to-one (see ``glTF.fbs`` next to this module).
Dictionaries that FlatBuffers cannot express (``extensions``, ``extras``,
primitive ``attributes``) are embedded as FlexBuffers.

``deserialize_root`` turns the FlatBuffer back into the equivalent plain
glTF dictionary so the rest of the pipeline can treat Odin files like any
other glTF document.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flatbuffers import flexbuffers

from . import gltf_generated as fb

# ---------------------------------------------------------------------------
# Schema description
#
# Each table is a dict mapping the glTF (camelCase) field name to a field
# spec. Field specs are:
#   Scalar(kind, unset)     scalar; dropped from the output when == unset
#   Str                     UTF-8 string
#   Flex                    FlexBuffer blob decoded to Python objects
#   Enum(names, default)    integer enum rendered as its glTF string name
#   Vec(kind)               vector of numbers ("int"/"float") or strings
#   [table]                 vector of tables
#   table                   nested table
# ---------------------------------------------------------------------------


class Scalar:
    __slots__ = ("kind", "unset")

    def __init__(self, kind: str = "int", unset: Any = None):
        self.kind = kind
        self.unset = unset


class Str:
    pass


class Flex:
    pass


class Enum:
    __slots__ = ("names", "default")

    def __init__(self, names: tuple[str, ...], default: str | None = None):
        self.names = names
        self.default = default


class Vec:
    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind


ID = Scalar("int", -1)
INT0 = Scalar("int", 0)
FLOAT = Scalar("float", None)
FLOAT0 = Scalar("float", 0.0)
BOOL_FALSE = Scalar("bool", False)

_TEXTURE_INFO = {"index": ID, "texCoord": INT0, "extensions": Flex(), "extras": Flex()}

_SPARSE_INDICES = {
    "bufferView": ID,
    "byteOffset": INT0,
    "componentType": Scalar("int"),
    "extensions": Flex(),
    "extras": Flex(),
}
_SPARSE_VALUES = {
    "bufferView": Scalar("int"),
    "byteOffset": INT0,
    "extensions": Flex(),
    "extras": Flex(),
}
_SPARSE = {
    "count": Scalar("int"),
    "indices": _SPARSE_INDICES,
    "values": _SPARSE_VALUES,
    "extensions": Flex(),
    "extras": Flex(),
}

ACCESSOR = {
    "bufferView": ID,
    "byteOffset": INT0,
    "componentType": Scalar("int"),
    "normalized": BOOL_FALSE,
    "count": Scalar("int"),
    "type": Enum(("SCALAR", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4")),
    "max": Vec("float"),
    "min": Vec("float"),
    "sparse": _SPARSE,
    "name": Str(),
    "extensions": Flex(),
    "extras": Flex(),
}

ANIMATION = {
    "name": Str(),
    "channels": [
        {
            "sampler": Scalar("int"),
            "target": {
                "node": ID,
                "path": Enum(("translation", "rotation", "scale", "weights")),
                "extensions": Flex(),
                "extras": Flex(),
            },
            "extensions": Flex(),
            "extras": Flex(),
        }
    ],
    "samplers": [
        {
            "input": Scalar("int"),
            "interpolation": Enum(("LINEAR", "STEP", "CUBICSPLINE"), default="LINEAR"),
            "output": Scalar("int"),
            "extensions": Flex(),
            "extras": Flex(),
        }
    ],
    "extensions": Flex(),
    "extras": Flex(),
}

ASSET = {
    "copyright": Str(),
    "generator": Str(),
    "version": Str(),
    "minVersion": Str(),
    "extensions": Flex(),
    "extras": Flex(),
}

BUFFER = {
    "name": Str(),
    "uri": Str(),
    "byteLength": Scalar("int"),
    "extensions": Flex(),
    "extras": Flex(),
}

BUFFER_VIEW = {
    "name": Str(),
    "buffer": ID,
    "byteOffset": INT0,
    "byteLength": Scalar("int"),
    "byteStride": INT0,
    "extensions": Flex(),
    "extras": Flex(),
}

CAMERA = {
    "name": Str(),
    "type": Enum(("orthographic", "perspective")),
    "orthographic": {
        "xmag": FLOAT,
        "ymag": FLOAT,
        "zfar": FLOAT,
        "znear": FLOAT,
        "extensions": Flex(),
        "extras": Flex(),
    },
    "perspective": {
        "aspectRatio": FLOAT0,
        "yfov": FLOAT,
        "zfar": FLOAT0,
        "znear": FLOAT,
        "extensions": Flex(),
        "extras": Flex(),
    },
    "extensions": Flex(),
    "extras": Flex(),
}

IMAGE = {
    "name": Str(),
    "uri": Str(),
    "mimeType": Str(),
    "bufferView": ID,
    "extensions": Flex(),
    "extras": Flex(),
}

MATERIAL = {
    "name": Str(),
    "pbrMetallicRoughness": {
        "baseColorFactor": Vec("float"),
        "baseColorTexture": _TEXTURE_INFO,
        "metallicFactor": Scalar("float", 1.0),
        "roughnessFactor": Scalar("float", 1.0),
        "metallicRoughnessTexture": _TEXTURE_INFO,
        "extensions": Flex(),
        "extras": Flex(),
    },
    "normalTexture": {"scale": Scalar("float", 1.0), **_TEXTURE_INFO},
    "occlusionTexture": {"strength": Scalar("float", 1.0), **_TEXTURE_INFO},
    "emissiveTexture": _TEXTURE_INFO,
    "emissiveFactor": Vec("float"),
    "doubleSided": BOOL_FALSE,
    "alphaCutoff": Scalar("float", 0.5),
    "alphaMode": Enum(("OPAQUE", "MASK", "BLEND"), default="OPAQUE"),
    "extensions": Flex(),
    "extras": Flex(),
}

MESH = {
    "name": Str(),
    "primitives": [
        {
            "attributes": Flex(),
            "indices": ID,
            "material": ID,
            "mode": Scalar("int", 4),
            "targets": Flex(),
            "extensions": Flex(),
            "extras": Flex(),
        }
    ],
    "weights": Vec("float"),
    "extensions": Flex(),
    "extras": Flex(),
}

NODE = {
    "name": Str(),
    "camera": ID,
    "children": Vec("int"),
    "skin": ID,
    "matrix": Vec("float"),
    "mesh": ID,
    "rotation": Vec("float"),
    "scale": Vec("float"),
    "translation": Vec("float"),
    "weights": Vec("float"),
    "extensions": Flex(),
    "extras": Flex(),
}

SAMPLER = {
    "name": Str(),
    "magFilter": INT0,
    "minFilter": INT0,
    "wrapS": Scalar("int", 10497),
    "wrapT": Scalar("int", 10497),
    "extensions": Flex(),
    "extras": Flex(),
}

SCENE = {"name": Str(), "nodes": Vec("int"), "extensions": Flex(), "extras": Flex()}

SKIN = {
    "name": Str(),
    "inverseBindMatrices": ID,
    "skeleton": ID,
    "joints": Vec("int"),
    "extensions": Flex(),
    "extras": Flex(),
}

TEXTURE = {"name": Str(), "sampler": ID, "source": ID, "extensions": Flex(), "extras": Flex()}

ROOT = {
    "asset": ASSET,
    "extensionsUsed": Vec("str"),
    "extensionsRequired": Vec("str"),
    "accessors": [ACCESSOR],
    "animations": [ANIMATION],
    "buffers": [BUFFER],
    "bufferViews": [BUFFER_VIEW],
    "cameras": [CAMERA],
    "images": [IMAGE],
    "materials": [MATERIAL],
    "meshes": [MESH],
    "nodes": [NODE],
    "samplers": [SAMPLER],
    "scene": ID,
    "scenes": [SCENE],
    "skins": [SKIN],
    "textures": [TEXTURE],
    "extensions": Flex(),
    "extras": Flex(),
}


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def _pascal(name: str) -> str:
    return name[0].upper() + name[1:]


def _clean(value: Any) -> Any:
    """Drop nulls and tidy numeric types in FlexBuffer-decoded data."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value if v is not None]
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def _decode_flex(table: Any, getter: str) -> Any:
    raw = getattr(table, f"{getter}AsNumpy")()
    if isinstance(raw, int):  # generated code returns 0 for absent vectors
        return None
    if len(raw) == 0:
        return None
    return _clean(flexbuffers.Loads(bytearray(raw)))


def _decode_vector(table: Any, getter: str, kind: str) -> list | None:
    if kind == "str":
        count = getattr(table, f"{getter}Length")()
        if count == 0:
            return None
        return [bytes(getattr(table, getter)(i)).decode("utf-8") for i in range(count)]

    raw = getattr(table, f"{getter}AsNumpy")()
    if isinstance(raw, int) or len(raw) == 0:
        return None
    values = np.asarray(raw).tolist()
    if kind == "float":
        return [round(float(v), 6) for v in values]
    return [int(v) for v in values]


def _decode_table(table: Any, schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in schema.items():
        getter = _pascal(key)

        if isinstance(spec, Scalar):
            value = getattr(table, getter)()
            if spec.kind == "bool":
                value = bool(value)
            elif spec.kind == "float":
                value = round(float(value), 6)
            else:
                value = int(value)
            if spec.unset is not None and value == spec.unset:
                continue
            out[key] = value

        elif isinstance(spec, Str):
            raw = getattr(table, getter)()
            if raw is not None:
                out[key] = bytes(raw).decode("utf-8")

        elif isinstance(spec, Flex):
            value = _decode_flex(table, getter)
            if value is not None:
                out[key] = value

        elif isinstance(spec, Enum):
            index = int(getattr(table, getter)())
            name = spec.names[index] if 0 <= index < len(spec.names) else str(index)
            if spec.default is not None and name == spec.default:
                continue
            out[key] = name

        elif isinstance(spec, Vec):
            value = _decode_vector(table, getter, spec.kind)
            if value is not None:
                out[key] = value

        elif isinstance(spec, list):
            count = getattr(table, f"{getter}Length")()
            if count:
                out[key] = [_decode_table(getattr(table, getter)(i), spec[0]) for i in range(count)]

        elif isinstance(spec, dict):
            nested = getattr(table, getter)()
            if nested is not None:
                out[key] = _decode_table(nested, spec)

        else:  # pragma: no cover - schema authoring error
            raise TypeError(f"bad schema spec for {key}: {spec!r}")

    return out


def _fix_primitive_targets(gltf: dict[str, Any]) -> None:
    # Supercell writes ``targets`` as an opaque FlexBuffer that is not a list
    # of attribute dictionaries. Keep it around as an extra for inspection.
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            targets = primitive.pop("targets", None)
            if targets is None:
                continue
            if isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                primitive["targets"] = targets
            else:
                primitive.setdefault("extras", {})["scTargets"] = targets


def deserialize_root(data: bytes) -> dict[str, Any]:
    """Decode a ``FLA2`` chunk into a glTF dictionary."""
    root = fb.Root.GetRootAs(bytearray(data), 0)
    gltf = _decode_table(root, ROOT)
    _fix_primitive_targets(gltf)
    gltf.setdefault("asset", {}).setdefault("version", "2.0")
    return gltf
