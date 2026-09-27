from __future__ import annotations

import numpy as np

from bsx.gltf.accessors import buffer_view_bytes, read_accessor
from bsx.odin.formats import AttributeFormat, format_info
from bsx.odin.mesh import decode_attribute, decode_mesh_data_info, vertex_capacity
from bsx.odin.normalize import normalize


def test_format_table_sizes():
    assert format_info(AttributeFormat.Float3).byte_size == 12
    assert format_info(AttributeFormat.Short2Norm).normalized
    assert format_info(AttributeFormat.UByte4).components == 4
    assert format_info(AttributeFormat.Float4x4).byte_size == 64


def test_packed_bone_weights_unpack():
    packed = np.array([[2047 << 21 | 1023 << 10 | 0]], dtype=np.uint32).view(np.uint8)
    weights = decode_attribute(packed, "a_boneweights", AttributeFormat.UInt)
    assert weights.shape == (1, 4)
    x = 2047 * 0.0002442
    y = 1023 * 0.0002442
    np.testing.assert_allclose(weights[0], [1 - x - y, x, y, 0], rtol=1e-5)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_normalized_short_uv_and_byte_normal():
    uv_raw = np.array([[32767, -32767]], dtype=np.int16).view(np.uint8)
    uv = decode_attribute(uv_raw, "a_uv0", AttributeFormat.Short2Norm)
    np.testing.assert_allclose(uv, [[1.0, -1.0]])

    normal_raw = np.array([[0, 0, 127, 55]], dtype=np.int8).view(np.uint8)
    normal = decode_attribute(normal_raw, "a_normal", AttributeFormat.Byte4Norm)
    assert normal.shape == (1, 3)
    np.testing.assert_allclose(normal, [[0, 0, 1]], atol=1e-6)


def test_joints_keep_integer_dtype():
    raw = np.array([[3, 2, 1, 0]], dtype=np.uint8)
    joints = decode_attribute(raw, "a_boneindex", AttributeFormat.UByte4)
    assert joints.dtype == np.uint8
    assert joints.tolist() == [[3, 2, 1, 0]]


def test_decode_streams_from_synthetic_document(synthetic_odin_document):
    gltf = normalize(synthetic_odin_document.gltf)
    binary = synthetic_odin_document.binary
    odin = gltf["extensions"]["SC_odin_format"]
    view = buffer_view_bytes(gltf, binary, odin["bufferView"])
    info = odin["meshDataInfos"][0]

    assert vertex_capacity(len(view), info) == 5
    attributes = decode_mesh_data_info(view, info, 5)
    assert set(attributes) == {"POSITION", "JOINTS_0", "WEIGHTS_0", "NORMAL", "TEXCOORD_0"}
    np.testing.assert_allclose(attributes["POSITION"][2], [1, 1, 0])
    np.testing.assert_allclose(attributes["TEXCOORD_0"][2], [1, 1], atol=1e-4)
    np.testing.assert_allclose(attributes["NORMAL"], [[0, 0, 1]] * 5)
    assert attributes["JOINTS_0"][2].tolist() == [2, 1, 0, 0]
    np.testing.assert_allclose(attributes["WEIGHTS_0"].sum(axis=1), 1.0, atol=1e-6)

    # componentType flag bits were masked by normalize()
    ibm = read_accessor(gltf, binary, 2)
    assert ibm.shape == (3, 16)
    assert ibm[1, 12] == -1.0


def test_normalize_repairs_structure(synthetic_odin_document):
    gltf = normalize(synthetic_odin_document.gltf)
    assert gltf["nodes"][0]["children"] == [1, 4]
    assert gltf["nodes"][1]["children"] == [2]
    assert "scale" not in gltf["nodes"][0]  # bogus [0,0,0] root scale removed
    assert gltf["scenes"] == [{"nodes": [0]}]
    assert gltf["scene"] == 0
    assert [m["name"] for m in gltf["materials"]] == ["body_mat", "glass_mat"]
    assert gltf["nodes"][4]["skin"] == 0  # skinned mesh node received the only skin
    # the input document is untouched
    assert "children" not in synthetic_odin_document.gltf["nodes"][0]
