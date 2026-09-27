"""Look up character textures in Brawl Stars' ``csv_logic/skins.csv``.

Odin geometry files do not name the character's colour texture; the game
resolves it per skin through this table (``DiffuseTexture`` /
``SpecularTexture`` columns, file names relative to ``sc3d/``).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class SkinInfo:
    name: str
    diffuse_texture: str | None
    specular_texture: str | None
    materials_file: str | None
    custom_shader: str | None


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def parse_skins_csv(text: str) -> list[SkinInfo]:
    """Parse the Supercell CSV (row 1 = header, row 2 = column types)."""
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return []
    header = rows[0]
    index = {name: i for i, name in enumerate(header)}

    def get(row: list[str], column: str) -> str | None:
        i = index.get(column)
        return _clean(row[i]) if i is not None and i < len(row) else None

    skins: list[SkinInfo] = []
    for row in rows[2:]:
        name = get(row, "Name")
        if not name:
            continue
        skins.append(
            SkinInfo(
                name=name,
                diffuse_texture=get(row, "DiffuseTexture"),
                specular_texture=get(row, "SpecularTexture"),
                materials_file=get(row, "MaterialsFile"),
                custom_shader=get(row, "CustomShader"),
            )
        )
    return skins


def find_skin(skins: list[SkinInfo], name: str) -> SkinInfo | None:
    lowered = name.lower()
    for skin in skins:
        if skin.name.lower() == lowered:
            return skin
    return None


def search_skins(skins: list[SkinInfo], needle: str) -> list[SkinInfo]:
    lowered = needle.lower()
    return [
        skin
        for skin in skins
        if lowered in skin.name.lower() or lowered in (skin.diffuse_texture or "").lower()
    ]
