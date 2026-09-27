"""Resolve texture names referenced by Odin materials to PNG bytes."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .sctx import SctxError, sctx_to_png

log = logging.getLogger(__name__)

Fetcher = Callable[[str], bytes | None]

_IMAGE_SUFFIXES = (".sctx", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp")


@dataclass
class ResolvedTexture:
    name: str  # cleaned texture name as referenced by the material
    source: str  # where the bytes came from (path or remote asset)
    png: bytes
    repeat: bool


def split_texture_reference(reference: str) -> tuple[str, list[str]]:
    """``"diffuse_lightmap.png#repeat"`` -> ``("diffuse_lightmap.png", ["repeat"])``."""
    if "#" in reference:
        name, keywords = reference.split("#", 1)
        return name, [k for k in keywords.split("+") if k]
    return reference, []


def image_bytes_to_png(name: str, data: bytes) -> bytes:
    suffix = Path(name).suffix.lower()
    if suffix == ".sctx":
        return sctx_to_png(data)
    if suffix == ".png":
        return data
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGBA")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@dataclass
class TextureResolver:
    lookup_dirs: list[Path] = field(default_factory=list)
    fetcher: Fetcher | None = None
    remote_prefix: str = "sc3d/"
    _cache: dict[str, ResolvedTexture | None] = field(default_factory=dict)

    def candidates(self, name: str) -> list[str]:
        path = Path(name)
        stem = path.stem
        names = [path.name]
        for suffix in _IMAGE_SUFFIXES:
            candidate = f"{stem}{suffix}"
            if candidate not in names:
                names.append(candidate)
        return names

    def _load_local(self, name: str) -> tuple[str, bytes] | None:
        direct = Path(name)
        if direct.is_file():
            return str(direct), direct.read_bytes()
        for directory in self.lookup_dirs:
            for candidate in self.candidates(name):
                path = directory / candidate
                if path.is_file():
                    return str(path), path.read_bytes()
        return None

    def _load_remote(self, name: str) -> tuple[str, bytes] | None:
        if self.fetcher is None:
            return None
        for candidate in self.candidates(name):
            remote = f"{self.remote_prefix}{candidate}"
            try:
                data = self.fetcher(remote)
            except Exception as exc:  # network errors are reported, not fatal
                log.warning("failed to fetch %s: %s", remote, exc)
                continue
            if data:
                return remote, data
        return None

    def resolve_file(self, path: str | Path) -> ResolvedTexture:
        """Resolve an explicit local file (user override)."""
        path = Path(path)
        return ResolvedTexture(
            name=path.name,
            source=str(path),
            png=image_bytes_to_png(path.name, path.read_bytes()),
            repeat=True,
        )

    def resolve(self, reference: str) -> ResolvedTexture | None:
        name, keywords = split_texture_reference(reference)
        if not name or name in (".", ""):
            return None
        if name in self._cache:
            return self._cache[name]

        found = self._load_local(name) or self._load_remote(name)
        result: ResolvedTexture | None = None
        if found is not None:
            source, data = found
            try:
                result = ResolvedTexture(
                    name=name,
                    source=source,
                    png=image_bytes_to_png(source, data),
                    repeat="repeat" in keywords,
                )
            except (SctxError, OSError, ValueError) as exc:
                log.warning("failed to decode texture %s: %s", source, exc)
        else:
            log.warning("texture not found: %s", name)

        self._cache[name] = result
        return result
