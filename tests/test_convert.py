from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from conftest import make_rle_animation_document
from PIL import Image

from bsx.cli import main
from bsx.convert import AnimationInput, ConvertOptions, convert
from bsx.glb import read_glb, write_glb
from bsx.gltf.accessors import read_accessor
from bsx.textures.resolver import TextureResolver


def _png(color) -> bytes:
    out = io.BytesIO()
    Image.new("RGBA", (2, 2), color).save(out, format="PNG")
    return out.getvalue()


def test_convert_produces_standard_gltf(synthetic_odin_document, tmp_path):
    (tmp_path / "body.png").write_bytes(_png((255, 0, 0, 255)))
    options = ConvertOptions(resolver=TextureResolver(lookup_dirs=[tmp_path]), unlit=True)
    result = convert(synthetic_odin_document, options)
    gltf, binary = result.gltf, result.binary

    assert result.warnings == []
    assert "SC_odin_format" not in json.dumps(gltf.get("extensionsUsed", []))
    assert gltf["extensionsUsed"] == ["KHR_materials_unlit"]
    assert gltf["scenes"] == [{"nodes": [0]}]
    assert gltf["nodes"][0]["children"] == [1, 4]
    assert "extensions" not in gltf["nodes"][1]

    mesh = gltf["meshes"][0]
    assert len(mesh["primitives"]) == 2
    attrs = mesh["primitives"][0]["attributes"]
    assert set(attrs) == {"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    # both primitives share the compacted vertex set
    assert mesh["primitives"][1]["attributes"] == attrs
    # only the 4 referenced vertices survive compaction (vertex 4 is unused)
    position = gltf["accessors"][attrs["POSITION"]]
    assert position["count"] == 4
    assert position["min"] == [0.0, 0.0, 0.0] and position["max"] == [1.0, 1.0, 0.0]
    assert mesh["extras"]["bounds"] == [[0, 0, 0], [1, 1, 0]]

    indices_a = read_accessor(gltf, binary, mesh["primitives"][0]["indices"]).reshape(-1)
    indices_b = read_accessor(gltf, binary, mesh["primitives"][1]["indices"]).reshape(-1)
    assert indices_a.tolist() == [0, 1, 2] and indices_b.tolist() == [0, 2, 3]

    skin = gltf["skins"][0]
    assert skin["joints"] == [1, 2, 3]
    ibm = read_accessor(gltf, binary, skin["inverseBindMatrices"])
    assert ibm.shape == (3, 16) and ibm[1, 12] == -1.0
    assert gltf["nodes"][4] == {"name": "quadGeo", "mesh": 0, "skin": 0}

    body, glass = gltf["materials"]
    assert body["name"] == "body_mat"
    assert body["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0}
    assert body["extensions"] == {"KHR_materials_unlit": {}}
    assert body["extras"]["SC_shader"]["constants"] == ["DIFFUSE"]
    assert "alphaMode" not in body
    assert glass["alphaMode"] == "BLEND"
    assert glass["pbrMetallicRoughness"]["baseColorFactor"] == [0.2, 0.4, 0.6, 0.5]

    assert gltf["images"][0]["mimeType"] == "image/png"
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    png = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    assert Image.open(io.BytesIO(png)).getpixel((0, 0)) == (255, 0, 0, 255)

    # the whole thing survives a GLB round trip
    doc = read_glb(write_glb(gltf, binary))
    assert doc.kind == "JSON" and not doc.is_odin


def test_convert_external_textures_and_overrides(synthetic_odin_document, tmp_path):
    override = tmp_path / "skin.png"
    override.write_bytes(_png((0, 255, 0, 255)))
    resolver = TextureResolver(lookup_dirs=[tmp_path])
    options = ConvertOptions(
        resolver=resolver,
        texture_overrides={"diffuseTex": resolver.resolve_file(override)},
        embed_textures=False,
    )
    result = convert(synthetic_odin_document, options)
    assert list(result.external_images) == ["skin.png"]
    assert result.gltf["images"][0]["uri"] == "skin.png"
    # the override applies to every material, including ones without a diffuse slot
    for material in result.gltf["materials"]:
        assert material["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0}


def test_convert_with_retargeted_animation(synthetic_odin_document):
    source_nodes = [
        {"name": "CHARACTER"},
        {"name": "root_bone", "extensions": {"SC_odin_format": {"parent": 0}}},
        {
            "name": "helper",
            "translation": [0, 0, 0],
            "extensions": {"SC_odin_format": {"parent": 1}},
        },
        {
            "name": "child_bone",
            "translation": [1, 0, 0],
            "extensions": {"SC_odin_format": {"parent": 2}},
        },
    ]
    specs = [
        {
            "node": 2,
            "flags": 4,
            "base": {"t": [0, 0, 0], "r": [0, 0, 0, 1], "s": [1, 1, 1], "tmul": 1.0, "smul": 0.0},
            "stream": [3, 0, 0, 0, 0, 1, 0, 0, 2, 0],
        }
    ]
    clip = make_rle_animation_document(source_nodes, specs, 3)
    result = convert(synthetic_odin_document, ConvertOptions(), [AnimationInput("wave", clip)])

    assert result.warnings == []
    animations = result.gltf["animations"]
    assert len(animations) == 1 and animations[0]["name"] == "wave"
    channels = animations[0]["channels"]
    # only child_bone (node 2 in the target) actually moves
    moving = {(c["target"]["node"], c["target"]["path"]) for c in channels}
    assert moving == {(2, "translation")}
    sampler = animations[0]["samplers"][channels[0]["sampler"]]
    times = read_accessor(result.gltf, result.binary, sampler["input"]).reshape(-1)
    values = read_accessor(result.gltf, result.binary, sampler["output"])
    np.testing.assert_allclose(times, [0, 1 / 30, 2 / 30], rtol=1e-6)
    np.testing.assert_allclose(values, [[1, 0, 0], [1, 1, 0], [1, 2, 0]], atol=1e-6)


def test_cli_inspect_and_convert(synthetic_odin_document, tmp_path, monkeypatch):
    from conftest import make_glb_bytes

    source = tmp_path / "quad_geo.glb"
    source.write_bytes(
        make_glb_bytes(
            b"JSON",
            json.dumps(synthetic_odin_document.gltf).encode(),
            synthetic_odin_document.binary,
        )
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["inspect", str(source)])
    assert result.exit_code == 0, result.output
    assert "vertex layout 0" in result.output
    assert "body_mat" in result.output

    result = runner.invoke(main, ["convert", str(source), "-o", "out/quad.glb"])
    assert result.exit_code == 0, result.output
    assert "warning: texture not found: body.png" in result.output
    doc = read_glb((tmp_path / "out/quad.glb").read_bytes())
    assert doc.kind == "JSON"
    assert len(doc.gltf["meshes"]) == 1


@pytest.mark.skipif(
    not os.environ.get("BSX_NETWORK_TESTS"), reason="set BSX_NETWORK_TESTS=1 to hit the asset index"
)
def test_network_end_to_end(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "convert",
            "sc3d/shelly_geo.glb",
            "--anim",
            "sc3d/shelly_xmas_idle.glb",
            "--skin",
            "BanditGirlDefault",
            "--fetch",
            "-o",
            str(tmp_path / "shelly.glb"),
        ],
        env={"BSX_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.exit_code == 0, result.output
    doc = read_glb((tmp_path / "shelly.glb").read_bytes())
    assert len(doc.gltf["animations"]) == 1
    assert len(doc.gltf["images"]) == 1
    assert Path(tmp_path / "cache").exists()
