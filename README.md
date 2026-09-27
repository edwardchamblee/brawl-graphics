# bsx

Command line extractor for **Brawl Stars** (and other Supercell "Odin" engine) 3D assets.

`bsx` turns the game's proprietary `.glb` files into **standard glTF 2.0** that any viewer,
engine or DCC tool can open, decodes `.sctx` textures to PNG, merges the separately shipped
skeletal animation clips onto a character, and can pull everything straight from the public
[sc-workshop](https://github.com/sc-workshop) asset index. It runs headless, without Blender.

The format knowledge comes from reading
[Daniil-SV/gltf-Supercell-IO](https://github.com/Daniil-SV/gltf-Supercell-IO) (a Blender
extension) and [sc-workshop/SupercellTexture](https://github.com/sc-workshop/SupercellTexture);
the glTF FlatBuffer schema and its generated reader are vendored from the former, which is why
this project is GPL-3.0 as well.

> This is unofficial fan tooling and is not endorsed by Supercell. See Supercell's
> [Fan Content Policy](https://www.supercell.com/fan-content-policy).

## Install

Python 3.11+.

```bash
pip install .            # or: pip install -e ".[dev]" for development
bsx --help
```

Dependencies: `numpy`, `flatbuffers`, `requests`, `Pillow`, `zstandard`, `texture2ddecoder`
(ASTC decoding), `click`.

## Quick start

Convert Shelly with her default skin texture and three animation clips, downloading every
input from the asset index:

```bash
bsx convert sc3d/shelly_geo.glb \
    --anim sc3d/shelly_xmas_idle.glb \
    --anim sc3d/shelly_xmas_walk.glb \
    --anim sc3d/shelly_xmas_attack.glb \
    --skin BanditGirlDefault --fetch \
    -o shelly.glb
```

`shelly.glb` is a self-contained, validator-clean glTF 2.0 binary: 8 skinned meshes, 46 joints,
the decoded 512x512 PNG texture and three named animation clips.

Working with local files only:

```bash
bsx inspect shelly_geo.glb                              # what is in the file
bsx texture shelly_redux_tex.sctx                       # -> shelly_redux_tex.png
bsx convert shelly_geo.glb --diffuse shelly_redux_tex.sctx --anim shelly_idle.glb -o shelly.glb
```

## Commands

| Command | Purpose |
| --- | --- |
| `bsx inspect FILE [--json]` | Summarise an Odin/standard `.glb`: vertex layouts, materials, skins, animation, node tree. `--json` dumps the full decoded document. |
| `bsx convert INPUT [-o OUT]` | Convert to standard glTF 2.0. Options: `--anim FILE` (repeatable, retargets clips by bone name), `--diffuse FILE`, `--texture SLOT=FILE`, `--skin NAME` (looks the texture up in `skins.csv`), `--texture-dir DIR`, `--textures embed\|external`, `--unlit`, `--fetch`. |
| `bsx texture INPUT... [-o OUT] [--level N] [--info] [--api]` | Decode `.sctx` (ASTC / RGBA8, zstd) to PNG locally, or through the sc-workshop web API for unsupported pixel formats. |
| `bsx games` / `bsx versions` | List game servers (`BS`, `BSCN`, `HD`) and indexed game versions. |
| `bsx search PATTERN` | Search asset paths, e.g. `bsx search "colt .glb$"`. |
| `bsx fetch PATH...` | Download assets by exact path, e.g. `sc3d/colt_geo.glb`. |
| `bsx skins [QUERY]` | Browse `csv_logic/skins.csv`: which diffuse/specular texture each skin uses. |

All network commands accept `--game` and `--version`. Downloads made by `--fetch` are cached in
`~/.cache/bsx` (override with `BSX_CACHE_DIR`).

## What the converter does

Odin `.glb` files are GLB containers whose first chunk is a FlatBuffer (`FLA2`) instead of
JSON, with the interesting data hidden in the `SC_odin_format` extension. `bsx convert`:

1. decodes the FlatBuffer into a glTF dictionary (`bsx.flat`);
2. repairs the document (`bsx.odin.normalize`): masks Supercell's flag bits out of accessor
   `componentType`, rebuilds `children` from per-node `parent` links, synthesises `scenes`,
   removes bogus `[0,0,0]` root scales, promotes the extension materials;
3. de-interleaves the GPU vertex streams described by `meshDataInfos` (`bsx.odin.mesh`):
   positions, normals, UVs, colours, tangents, joint indices and the 11/11/10-bit packed bone
   weights become proper `POSITION`, `NORMAL`, `TEXCOORD_0`, `JOINTS_0`, `WEIGHTS_0`...
   accessors, compacted per mesh so only referenced vertices survive;
4. copies skins and inverse bind matrices (merging duplicate joints if present);
5. decodes the animation (`bsx.odin.animation`): raw, packed or RLE-packed int16 delta streams
   are expanded to per-frame TRS keyframes;
6. retargets external clips (`bsx.gltf.retarget`): animation rigs contain helper bones
   (`hips_s:SSC`, `torso`...) that geometry rigs lack, so the animation rig is evaluated in
   world space and every geometry bone with the same name receives the local transform that
   reproduces it;
7. maps the `uber` shader materials to PBR (`baseColorTexture`, blend/cutout alpha modes,
   double-sided; `--unlit` adds `KHR_materials_unlit`), keeping the original shader parameters in
   `extras.SC_shader`;
8. resolves textures: explicit overrides, then names referenced by the material, searched in
   the input directory, `--texture-dir`s and (with `--fetch`) the asset index; `.sctx` files are
   decoded to PNG and embedded or written next to the output.

Character colour textures are not referenced by the geometry file. The game picks them per
skin through `csv_logic/skins.csv`, which is what `--skin` reads (`bsx skins shelly` shows
the candidates).

## Library use

```python
from bsx.glb import read_glb_file, write_glb
from bsx.convert import convert, ConvertOptions, AnimationInput

geo = read_glb_file("shelly_geo.glb")
idle = read_glb_file("shelly_xmas_idle.glb")
result = convert(geo, ConvertOptions(), [AnimationInput("idle", idle)])
open("shelly.glb", "wb").write(write_glb(result.gltf, result.binary))
```

Lower level entry points: `bsx.flat.deserialize_root`, `bsx.odin.mesh.decode_mesh_data_info`,
`bsx.odin.animation.read_animation`, `bsx.textures.sctx.parse_sctx` / `sctx_to_png`,
`bsx.net.assets.AssetClient`.

## Limitations

- Lightmaps, specular, stencil and colorize maps are preserved as metadata but not translated
  into PBR inputs; the in-game look relies on Supercell's `uber` shader.
- Only ASTC and 8-bit uncompressed pixel formats are decoded locally; other `.sctx` formats can
  go through `bsx texture --api`.
- Legacy `.scw` (SC3D) models and `.sc` Flash textures are not supported.
- Sparse accessors and multi-buffer glTF inputs are not supported (Odin files use neither).

## Development

```bash
pip install -e ".[dev]"
pytest                          # unit tests on synthetic assets
BSX_NETWORK_TESTS=1 pytest      # also runs a live end-to-end download + convert
ruff check src tests && ruff format --check src tests
```

## License

GPL-3.0-or-later. `src/bsx/flat/gltf_generated.py` and `glTF.fbs` originate from
gltf-Supercell-IO (GPL-3.0-or-later).
