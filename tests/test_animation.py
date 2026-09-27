from __future__ import annotations

import numpy as np
import pytest
from conftest import make_rle_animation_document

from bsx.gltf.retarget import direct_tracks, retarget
from bsx.odin.animation import AnimationError, find_animation_descriptor, read_animation
from bsx.odin.normalize import normalize

HALF_TURN_Y = [0.0, 1.0, 0.0, 0.0]  # xyzw quaternion, 180 degrees around Y


def _source_nodes():
    return [
        {"name": "A"},
        {"name": "H", "translation": [0, 5, 0], "extensions": {"SC_odin_format": {"parent": 0}}},
        {"name": "B", "translation": [1, 0, 0], "extensions": {"SC_odin_format": {"parent": 1}}},
    ]


def _document(frame_count=4):
    specs = [
        {
            "node": 1,
            "flags": 4,  # translation
            "base": {"t": [0, 5, 0], "r": [0, 0, 0, 1], "s": [1, 1, 1], "tmul": 0.5, "smul": 0.0},
            # two literal frames then "repeat previous" twice
            "stream": [2, 0, 0, 0, 2, 4, 6, -2],
        },
        {
            "node": 2,
            "flags": 2,  # rotation
            "base": {"t": [1, 0, 0], "r": [0, 0, 0, 1], "s": [1, 1, 1], "tmul": 0.0, "smul": 0.0},
            "stream": [1, 0, 32767, 0, 0, -3],
        },
    ]
    return make_rle_animation_document(_source_nodes(), specs, frame_count)


def test_rle_packed_animation_decodes():
    doc = _document()
    gltf = normalize(doc.gltf)
    animation = read_animation(gltf, doc.binary, find_animation_descriptor(gltf))

    assert animation.frame_rate == 30.0
    assert animation.frame_count == 4
    assert pytest.approx(animation.duration) == 3 / 30

    helper, child = animation.tracks
    assert helper.node == 1 and helper.animated["translation"] and not helper.animated["rotation"]
    np.testing.assert_allclose(helper.translation, [[0, 5, 0], [1, 7, 3], [1, 7, 3], [1, 7, 3]])
    np.testing.assert_allclose(helper.rotation, [[0, 0, 0, 1]] * 4)

    assert child.animated["rotation"]
    np.testing.assert_allclose(child.rotation, [HALF_TURN_Y] * 4, atol=1e-6)
    np.testing.assert_allclose(child.translation, [[1, 0, 0]] * 4)


def test_rle_data_size_mismatch_is_rejected():
    doc = _document()
    doc.gltf["extensions"]["SC_odin_format"]["animation"]["packed"]["nodes"][0]["dataSize"] = 3
    gltf = normalize(doc.gltf)
    with pytest.raises(AnimationError):
        read_animation(gltf, doc.binary, find_animation_descriptor(gltf))


def test_direct_tracks_fill_unanimated_channels_with_rest():
    doc = _document()
    gltf = normalize(doc.gltf)
    animation = read_animation(gltf, doc.binary, find_animation_descriptor(gltf))
    tracks = direct_tracks(gltf["nodes"], animation)
    child = tracks[1]
    np.testing.assert_allclose(child.scale, np.ones((4, 3)))
    np.testing.assert_allclose(child.translation, [[1, 0, 0]] * 4)


def test_retarget_folds_helper_motion_into_child():
    """Target rig lacks the helper node H; B must inherit H's translation."""
    doc = _document()
    gltf = normalize(doc.gltf)
    animation = read_animation(gltf, doc.binary, find_animation_descriptor(gltf))

    target_nodes = [
        {"name": "A", "children": [1]},
        {"name": "B", "translation": [1, 5, 0]},
    ]
    tracks, unmatched = retarget(gltf["nodes"], animation, target_nodes)
    assert unmatched == ["H"]
    assert [t.node for t in tracks] == [0, 1]

    b = tracks[1]
    # world(B) = T(H) * T(B) in the source; the target parent A is identity
    np.testing.assert_allclose(b.translation[0], [1, 5, 0], atol=1e-6)
    np.testing.assert_allclose(b.translation[1], [2, 7, 3], atol=1e-6)
    np.testing.assert_allclose(b.translation[3], [2, 7, 3], atol=1e-6)
    np.testing.assert_allclose(np.abs(b.rotation), [np.abs(HALF_TURN_Y)] * 4, atol=1e-6)
    np.testing.assert_allclose(b.scale, np.ones((4, 3)), atol=1e-6)
