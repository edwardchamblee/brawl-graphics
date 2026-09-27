"""Repair the structural quirks of Odin glTF documents.

Odin files omit or misuse several glTF fields. This module rewrites a decoded
document so that generic glTF code can walk it:

* accessor ``componentType`` carries private flag bits in the upper half word;
* node hierarchy is stored as ``extensions.SC_odin_format.parent`` per node
  instead of ``children`` arrays;
* ``scenes``/``scene`` are frequently missing;
* root nodes may carry a ``[0, 0, 0]`` scale;
* materials live in ``extensions.SC_odin_format.materials``;
* nodes in animation-only files may reference meshes that do not exist.
"""

from __future__ import annotations

import copy
from typing import Any

from ..gltf.accessors import component_type_of

ODIN = "SC_odin_format"
SC_SHADER = "SC_shader"


def odin_extension(gltf: dict[str, Any]) -> dict[str, Any]:
    return (gltf.get("extensions") or {}).get(ODIN) or {}


def _fix_accessors(gltf: dict[str, Any]) -> None:
    for accessor in gltf.get("accessors", []):
        if "componentType" in accessor:
            accessor["componentType"] = component_type_of(accessor["componentType"])


def _rebuild_children(gltf: dict[str, Any]) -> None:
    nodes = gltf.get("nodes", [])
    has_parent_links = any(
        ((node.get("extensions") or {}).get(ODIN) or {}).get("parent") is not None for node in nodes
    )
    if not has_parent_links:
        return

    children: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        parent = ((node.get("extensions") or {}).get(ODIN) or {}).get("parent")
        if parent is None or parent == index or not (0 <= parent < len(nodes)):
            continue
        children.setdefault(parent, []).append(index)

    for index, node in enumerate(nodes):
        if index in children:
            node["children"] = children[index]
        else:
            node.pop("children", None)


def _fix_references(gltf: dict[str, Any]) -> None:
    mesh_count = len(gltf.get("meshes", []))
    skin_count = len(gltf.get("skins", []))
    camera_count = len(gltf.get("cameras", []))
    for node in gltf.get("nodes", []):
        if node.get("mesh") is not None and node["mesh"] >= mesh_count:
            node.pop("mesh")
        if node.get("skin") is not None and node["skin"] >= skin_count:
            node.pop("skin")
        if node.get("camera") is not None and node["camera"] >= camera_count:
            node.pop("camera")


def _fix_scenes(gltf: dict[str, Any]) -> list[int]:
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes")
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    else:
        referenced = {child for node in nodes for child in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in referenced]
        gltf["scenes"] = [{"nodes": roots}]
    gltf.setdefault("scene", 0)

    for root in roots:
        if root < len(nodes) and nodes[root].get("scale") == [0, 0, 0]:
            nodes[root].pop("scale")
    return roots


def _move_materials(gltf: dict[str, Any]) -> None:
    materials = odin_extension(gltf).get("materials")
    if not materials or gltf.get("materials"):
        return
    gltf["materials"] = [
        {"name": material.get("name", f"material_{i}"), "extensions": {SC_SHADER: material}}
        for i, material in enumerate(materials)
    ]


def _assign_missing_skins(gltf: dict[str, Any]) -> None:
    skins = gltf.get("skins", [])
    if len(skins) != 1:
        return
    infos = odin_extension(gltf).get("meshDataInfos") or []

    def mesh_is_skinned(mesh: dict[str, Any]) -> bool:
        for primitive in mesh.get("primitives", []):
            if "JOINTS_0" in (primitive.get("attributes") or {}):
                return True
            index = ((primitive.get("extensions") or {}).get(ODIN) or {}).get("meshDataInfoIndex")
            if index is not None and index < len(infos):
                for descriptor in infos[index].get("vertexDescriptors", []):
                    if any(a.get("name") == "a_boneindex" for a in descriptor.get("attributes", [])):
                        return True
        return False

    meshes = gltf.get("meshes", [])
    for node in gltf.get("nodes", []):
        mesh = node.get("mesh")
        if mesh is not None and node.get("skin") is None and mesh_is_skinned(meshes[mesh]):
            node["skin"] = 0


def normalize(gltf: dict[str, Any]) -> dict[str, Any]:
    """Return a repaired deep copy of an Odin glTF document."""
    gltf = copy.deepcopy(gltf)
    for key in ("nodes", "meshes", "accessors", "bufferViews", "skins", "materials"):
        gltf.setdefault(key, [])
    _fix_accessors(gltf)
    _rebuild_children(gltf)
    _fix_references(gltf)
    _fix_scenes(gltf)
    _move_materials(gltf)
    _assign_missing_skins(gltf)
    return gltf
