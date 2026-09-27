from __future__ import annotations

import numpy as np
import pytest
from conftest import build_flat_root, fb, flex, make_glb_bytes

from bsx.flat import deserialize_root
from bsx.glb import GlbError, read_glb, write_glb


def _build_root(builder):
    version = builder.CreateString("2.0")
    fb.AssetStart(builder)
    fb.AssetAddVersion(builder, version)
    asset = fb.AssetEnd(builder)

    name = builder.CreateString("hips")
    node_ext = flex(builder, {"SC_odin_format": {"parent": 0}})
    fb.NodeStartTranslationVector(builder, 3)
    for value in reversed([1.5, 2.0, -3.0]):
        builder.PrependFloat32(value)
    translation = builder.EndVector()
    fb.NodeStart(builder)
    fb.NodeAddName(builder, name)
    fb.NodeAddTranslation(builder, translation)
    fb.NodeAddExtensions(builder, node_ext)
    node_a = fb.NodeEnd(builder)

    root_name = builder.CreateString("CHARACTER")
    fb.NodeStart(builder)
    fb.NodeAddName(builder, root_name)
    node_b = fb.NodeEnd(builder)

    fb.RootStartNodesVector(builder, 2)
    builder.PrependUOffsetTRelative(node_a)
    builder.PrependUOffsetTRelative(node_b)
    nodes = builder.EndVector()

    fb.AccessorStart(builder)
    fb.AccessorAddBufferView(builder, 0)
    fb.AccessorAddComponentType(builder, 0x10000 | 5123)
    fb.AccessorAddCount(builder, 12)
    fb.AccessorAddType(builder, 0)  # SCALAR
    accessor = fb.AccessorEnd(builder)
    fb.RootStartAccessorsVector(builder, 1)
    builder.PrependUOffsetTRelative(accessor)
    accessors = builder.EndVector()

    ext_name = builder.CreateString("SC_odin_format")
    fb.RootStartExtensionsRequiredVector(builder, 1)
    builder.PrependUOffsetTRelative(ext_name)
    required = builder.EndVector()

    root_ext = flex(builder, {"SC_odin_format": {"materials": [{"name": "m", "blendMode": 4}]}})

    fb.RootStart(builder)
    fb.RootAddAsset(builder, asset)
    fb.RootAddNodes(builder, nodes)
    fb.RootAddAccessors(builder, accessors)
    fb.RootAddExtensionsRequired(builder, required)
    fb.RootAddExtensions(builder, root_ext)
    return fb.RootEnd(builder)


def test_deserialize_flat_root():
    gltf = deserialize_root(build_flat_root(_build_root))
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["extensionsRequired"] == ["SC_odin_format"]
    assert gltf["nodes"][0] == {"name": "CHARACTER"}
    assert gltf["nodes"][1]["name"] == "hips"
    assert gltf["nodes"][1]["translation"] == [1.5, 2.0, -3.0]
    assert gltf["nodes"][1]["extensions"] == {"SC_odin_format": {"parent": 0}}
    accessor = gltf["accessors"][0]
    assert accessor["type"] == "SCALAR"
    assert accessor["count"] == 12
    assert accessor["componentType"] == 0x10000 | 5123
    assert "byteOffset" not in accessor  # default dropped
    assert gltf["extensions"]["SC_odin_format"]["materials"][0]["name"] == "m"


def test_read_glb_fla2_and_json_roundtrip():
    flat = build_flat_root(_build_root)
    doc = read_glb(make_glb_bytes(b"FLA2", flat, b"\x01\x02\x03"))
    assert doc.kind == "FLA2"
    assert doc.is_odin
    assert doc.binary[:3] == b"\x01\x02\x03"

    written = write_glb({"asset": {"version": "2.0"}, "nodes": [{"name": "x"}]}, b"abcde")
    back = read_glb(written)
    assert back.kind == "JSON"
    assert not back.is_odin
    assert back.gltf["nodes"][0]["name"] == "x"
    assert back.binary[:5] == b"abcde"
    assert len(written) % 4 == 0


def test_read_glb_rejects_garbage():
    with pytest.raises(GlbError):
        read_glb(b"nope" + b"\x00" * 20)
    with pytest.raises(GlbError):
        read_glb(make_glb_bytes(b"XXXX", b"{}", b""))


def test_write_glb_without_binary():
    data = write_glb({"asset": {"version": "2.0"}}, b"")
    doc = read_glb(data)
    assert doc.binary == b""
    assert np.frombuffer(data[8:12], dtype="<u4")[0] == len(data)
