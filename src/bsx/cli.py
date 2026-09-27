"""Command line interface for bsx."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from . import __version__
from .glb import GlbDocument, GlbError, read_glb_file, write_glb

log = logging.getLogger("bsx")


class _ClickHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        prefix = {"WARNING": "warning: ", "ERROR": "error: "}.get(record.levelname, "")
        click.echo(f"{prefix}{record.getMessage()}", err=True)


def _setup_logging(verbose: bool) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
    if not any(isinstance(h, _ClickHandler) for h in log.handlers):
        log.addHandler(_ClickHandler())


def _fail(message: str) -> None:
    raise click.ClickException(message)


def _load_glb(path: Path) -> GlbDocument:
    try:
        return read_glb_file(str(path))
    except (GlbError, OSError) as exc:
        _fail(f"{path}: {exc}")
    raise AssertionError  # unreachable


def _client():
    from .net.assets import AssetClient

    return AssetClient()


def _resolve_input(path: str, fetch: bool, game: str, version: str | None) -> Path:
    """Local file path, or an asset path (``sc3d/x.glb``) downloaded when --fetch is set."""
    local = Path(path)
    if local.is_file():
        return local
    if fetch:
        from .net.assets import AssetApiError, fetch_cached

        try:
            return fetch_cached(_client(), path, game, version)
        except AssetApiError as exc:
            _fail(f"could not fetch {path}: {exc}")
    _fail(f"{path}: file not found (use --fetch to download it from the asset index)")
    raise AssertionError  # unreachable


_game_option = click.option(
    "--game", default="BS", show_default=True, help="Game codename (see `bsx games`)."
)
_version_option = click.option(
    "--version", "game_version", default=None, help="Game version, e.g. 69.270.1 (default: latest)."
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="bsx")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(verbose: bool) -> None:
    """Extract Brawl Stars (Supercell Odin) models, animations and textures.

    Converts Odin ``.glb`` files to standard glTF 2.0, decodes ``.sctx``
    textures to PNG and can fetch assets straight from the public
    sc-workshop asset index.
    """
    _setup_logging(verbose)


# --------------------------------------------------------------------------- inspect


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Dump the decoded glTF document as JSON.")
@click.option("--raw", is_flag=True, help="With --json: skip Odin structure repairs.")
def inspect(file: Path, as_json: bool, raw: bool) -> None:
    """Summarise an Odin or standard .glb file."""
    from .odin.animation import find_animation_descriptor, read_animation
    from .odin.normalize import normalize, odin_extension

    doc = _load_glb(file)
    gltf = doc.gltf if raw else normalize(doc.gltf)

    if as_json:
        click.echo(json.dumps(gltf, indent=2))
        return

    odin = odin_extension(gltf)
    click.echo(
        f"{file.name}: {doc.kind} chunk, {len(doc.binary):,} bytes of binary data, odin={doc.is_odin}"
    )
    click.echo(
        f"  nodes={len(gltf.get('nodes', []))} meshes={len(gltf.get('meshes', []))} "
        f"skins={len(gltf.get('skins', []))} materials={len(gltf.get('materials', []))} "
        f"accessors={len(gltf.get('accessors', []))}"
    )

    infos = odin.get("meshDataInfos") or []
    for i, info in enumerate(infos):
        streams = []
        for descriptor in info.get("vertexDescriptors", []):
            names = ",".join(
                f"{a.get('name')}:{a.get('format')}" for a in descriptor.get("attributes", [])
            )
            streams.append(f"[stride {descriptor.get('stride')}: {names}]")
        click.echo(f"  vertex layout {i}: {' '.join(streams)}")

    for i, mesh in enumerate(gltf.get("meshes", [])):
        primitives = mesh.get("primitives", [])
        materials = sorted({p.get("material") for p in primitives if p.get("material") is not None})
        click.echo(
            f"  mesh {i} {mesh.get('name') or ''}: {len(primitives)} primitive(s), materials {materials}"
        )

    for i, material in enumerate(gltf.get("materials", [])):
        sc = (material.get("extensions") or {}).get("SC_shader")
        if sc is None:
            click.echo(f"  material {i}: {material.get('name')}")
            continue
        textures = (sc.get("variables") or {}).get("textures") or {}
        click.echo(
            f"  material {i}: {sc.get('name')} shader={sc.get('shader')} blend={sc.get('blendMode')} "
            f"constants={sc.get('constants')} textures={textures}"
        )

    for i, skin in enumerate(gltf.get("skins", [])):
        click.echo(f"  skin {i}: {len(skin.get('joints', []))} joints")

    descriptor = find_animation_descriptor(gltf)
    if descriptor is not None:
        try:
            animation = read_animation(gltf, doc.binary, descriptor)
            layout = (
                "rle-packed"
                if (descriptor.get("packed") or {}).get("uintAccessor") is not None
                else ("packed" if descriptor.get("packed") else "raw")
            )
            click.echo(
                f"  animation: {layout}, {animation.frame_count} frames @ {animation.frame_rate:g} fps "
                f"({animation.duration:.2f}s), {len(animation.tracks)} animated nodes"
            )
        except Exception as exc:  # report but keep inspecting
            click.echo(f"  animation: failed to decode ({exc})")

    roots = gltf.get("scenes", [{}])[gltf.get("scene", 0)].get("nodes", [])
    nodes = gltf.get("nodes", [])

    def tree(index: int, depth: int, budget: list[int]) -> None:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        node = nodes[index]
        tags = []
        if "mesh" in node:
            tags.append(f"mesh {node['mesh']}")
        if "skin" in node:
            tags.append(f"skin {node['skin']}")
        suffix = f"  ({', '.join(tags)})" if tags else ""
        click.echo(f"  {'  ' * depth}{node.get('name') or index}{suffix}")
        for child in node.get("children", []):
            tree(child, depth + 1, budget)

    budget = [40]
    click.echo("  hierarchy:")
    for root in roots:
        tree(root, 1, budget)
    if budget[0] <= 0:
        click.echo("  ... (truncated, use --json for everything)")


# --------------------------------------------------------------------------- network


@main.command()
def games() -> None:
    """List game servers known to the asset index."""
    for server in _client().servers():
        click.echo(f"{server.codename:6} {server.name}")


@main.command()
@_game_option
@click.option("--limit", default=20, show_default=True)
def versions(game: str, limit: int) -> None:
    """List indexed game versions (newest first)."""
    for item in _client().versions(game)[:limit]:
        click.echo(f"{item.version:12} {item.hash}")


@main.command()
@click.argument("pattern")
@_game_option
@_version_option
@click.option("--limit", default=50, show_default=True)
@click.option("--page", default=0, show_default=True)
def search(pattern: str, game: str, game_version: str | None, limit: int, page: int) -> None:
    """Search asset paths. PATTERN is regex tokens, e.g. "shelly .glb$"."""
    results = _client().search(pattern, game, game_version, limit, page)
    if not results:
        click.echo("no matches", err=True)
    for path in results:
        click.echo(path)


@main.command()
@click.argument("paths", nargs=-1, required=True)
@_game_option
@_version_option
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
def fetch(paths: tuple[str, ...], game: str, game_version: str | None, output: Path) -> None:
    """Download assets by exact path, e.g. sc3d/shelly_geo.glb."""
    from .net.assets import AssetApiError

    client = _client()
    for path in paths:
        try:
            data = client.download(path, game, game_version)
        except AssetApiError as exc:
            log.error("%s: %s", path, exc)
            continue
        target = output / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        click.echo(f"{target} ({len(data):,} bytes)")


# --------------------------------------------------------------------------- textures


@main.command()
@click.argument(
    "inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output PNG (single input) or directory.",
)
@click.option("--level", default=0, show_default=True, help="Mip level to export.")
@click.option("--info", is_flag=True, help="Print texture metadata instead of decoding.")
@click.option(
    "--api",
    is_flag=True,
    help="Convert through the sc-workshop web API instead of decoding locally.",
)
def texture(
    inputs: tuple[Path, ...], output: Path | None, level: int, info: bool, api: bool
) -> None:
    """Decode .sctx textures to PNG."""
    from .textures.sctx import SctxError, decode_rgba, parse_sctx

    if output is not None and len(inputs) > 1 and output.suffix.lower() == ".png":
        _fail("--output must be a directory when converting several files")

    for path in inputs:
        data = path.read_bytes()
        if info:
            try:
                tex = parse_sctx(data)
            except SctxError as exc:
                log.error("%s: %s", path, exc)
                continue
            levels = ", ".join(f"{m.width}x{m.height}" for m in tex.levels)
            click.echo(
                f"{path.name}: {tex.width}x{tex.height} {tex.format_name}, flags=0x{tex.flags:x}, mips: {levels}"
            )
            continue

        if output is None:
            target = path.with_suffix(".png")
        elif output.suffix.lower() == ".png":
            target = output
        else:
            target = output / path.with_suffix(".png").name
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            if api:
                from .net.neko import convert_sctx_to_png

                target.write_bytes(convert_sctx_to_png(path.name, data))
            else:
                decode_rgba(parse_sctx(data), level).save(target)
        except Exception as exc:
            log.error("%s: %s", path, exc)
            continue
        click.echo(str(target))


# --------------------------------------------------------------------------- skins


def _load_skins(csv_path: Path | None, game: str, version: str | None):
    from .skins import parse_skins_csv

    if csv_path is None:
        from .net.assets import fetch_cached

        csv_path = fetch_cached(_client(), "csv_logic/skins.csv", game, version)
    return parse_skins_csv(csv_path.read_text(encoding="utf-8", errors="replace"))


@main.command()
@click.argument("query", default="")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local skins.csv (downloaded when omitted).",
)
@_game_option
@_version_option
def skins(query: str, csv_path: Path | None, game: str, game_version: str | None) -> None:
    """Look up skins.csv: which texture belongs to which skin."""
    from .skins import search_skins

    entries = _load_skins(csv_path, game, game_version)
    matches = search_skins(entries, query) if query else entries
    if not matches:
        click.echo("no matching skins", err=True)
    for skin in matches:
        extra = f"  materials={skin.materials_file}" if skin.materials_file else ""
        click.echo(
            f"{skin.name:40} diffuse={skin.diffuse_texture or '-':40} specular={skin.specular_texture or '-'}{extra}"
        )


# --------------------------------------------------------------------------- convert


@main.command()
@click.argument("input_path", metavar="INPUT")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output .glb (default: <input>.converted.glb).",
)
@click.option(
    "--anim",
    "anims",
    multiple=True,
    metavar="FILE",
    help="Odin animation .glb to retarget onto INPUT (repeatable).",
)
@click.option(
    "--diffuse",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Diffuse texture (.sctx/.png) applied to every material.",
)
@click.option(
    "--texture",
    "texture_slots",
    multiple=True,
    metavar="SLOT=FILE",
    help="Override a material texture slot, e.g. emissionTex=glow.png.",
)
@click.option(
    "--skin", "skin_name", help="Skin name from skins.csv whose DiffuseTexture should be used."
)
@click.option(
    "--skins-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local skins.csv for --skin.",
)
@click.option(
    "--texture-dir",
    "texture_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Extra directories searched for textures (repeatable).",
)
@click.option(
    "--fetch", is_flag=True, help="Download missing inputs/textures from the asset index."
)
@_game_option
@_version_option
@click.option(
    "--textures",
    "texture_mode",
    type=click.Choice(["embed", "external"]),
    default="embed",
    show_default=True,
    help="Embed PNGs in the .glb or write them next to it.",
)
@click.option(
    "--unlit",
    is_flag=True,
    help="Tag materials with KHR_materials_unlit (closest to the in-game look).",
)
def convert(
    input_path: str,
    output: Path | None,
    anims: tuple[str, ...],
    diffuse: Path | None,
    texture_slots: tuple[str, ...],
    skin_name: str | None,
    skins_csv: Path | None,
    texture_dirs: tuple[Path, ...],
    fetch: bool,
    game: str,
    game_version: str | None,
    texture_mode: str,
    unlit: bool,
) -> None:
    """Convert an Odin .glb to standard glTF 2.0 (mesh, skin, textures, animations).

    INPUT and --anim accept local files, or asset paths such as
    sc3d/shelly_geo.glb together with --fetch.
    """
    from .convert import AnimationInput, ConvertOptions
    from .convert import convert as run_convert
    from .skins import find_skin
    from .textures.resolver import TextureResolver

    source = _resolve_input(input_path, fetch, game, game_version)
    document = _load_glb(source)

    fetcher = None
    if fetch:
        client = _client()

        def fetcher(remote: str, _client=client) -> bytes | None:
            from .net.assets import AssetApiError

            try:
                return _client.download(remote, game, game_version)
            except AssetApiError:
                return None

    lookup_dirs = [source.parent, Path.cwd(), *texture_dirs]
    resolver = TextureResolver(lookup_dirs=lookup_dirs, fetcher=fetcher)

    overrides = {}
    for spec in texture_slots:
        if "=" not in spec:
            _fail(f"--texture expects SLOT=FILE, got {spec!r}")
        slot, file = spec.split("=", 1)
        overrides[slot.strip()] = resolver.resolve_file(file)
    if diffuse is not None:
        overrides["diffuseTex"] = resolver.resolve_file(diffuse)

    if skin_name:
        entries = _load_skins(skins_csv, game, game_version)
        skin = find_skin(entries, skin_name)
        if skin is None:
            _fail(f"skin {skin_name!r} not found in skins.csv (try `bsx skins {skin_name}`)")
        assert skin is not None
        if not skin.diffuse_texture:
            log.warning("skin %s has no DiffuseTexture entry", skin.name)
        else:
            resolved = resolver.resolve(skin.diffuse_texture)
            if resolved is None:
                _fail(
                    f"texture {skin.diffuse_texture} for skin {skin.name} was not found locally"
                    + ("" if fetch else "; add --fetch to download it")
                )
            assert resolved is not None
            overrides.setdefault("diffuseTex", resolved)

    clips = []
    for anim in anims:
        path = _resolve_input(anim, fetch, game, game_version)
        clips.append(AnimationInput(name=path.stem, document=_load_glb(path)))

    options = ConvertOptions(
        resolver=resolver,
        texture_overrides=overrides,
        embed_textures=texture_mode == "embed",
        unlit=unlit,
        source_name=source.name,
    )
    try:
        result = run_convert(document, options, clips, own_animation_name=source.stem)
    except Exception as exc:
        log.debug("conversion failed", exc_info=True)
        raise click.ClickException(f"conversion failed: {exc}") from exc

    for warning in result.warnings:
        log.warning(warning)

    if output is None:
        output = Path.cwd() / f"{source.stem}.converted.glb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(write_glb(result.gltf, result.binary))
    for name, png in result.external_images.items():
        (output.parent / name).write_bytes(png)

    gltf = result.gltf
    summary = (
        f"{output}: {len(gltf.get('meshes', []))} meshes, {len(gltf.get('materials', []))} materials, "
        f"{len(gltf.get('skins', []))} skins, {len(gltf.get('animations', []))} animations, "
        f"{len(gltf.get('images', []))} images"
    )
    click.echo(summary)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
