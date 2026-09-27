"""Convert Supercell Odin glTF documents into standard glTF 2.0."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .glb import GlbDocument
from .gltf.accessors import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, BufferBuilder, buffer_view_bytes, read_accessor
from .gltf.retarget import RetargetedTrack, direct_tracks, retarget
from .gltf.transform import node_trs
from .odin.animation import find_animation_descriptor, read_animation
from .odin.mesh import decode_mesh_data_info, vertex_capacity
from .odin.normalize import ODIN, SC_SHADER, normalize, odin_extension
from .textures.resolver import ResolvedTexture, TextureResolver

log = logging.getLogger(__name__)

NODE_KEYS = ("name", "children", "translation", "rotation", "scale", "matrix", "mesh", "skin", "camera", "extras")

# Odin "uber" shader blend modes
BLEND_OPAQUE = 4

# Texture slot names used by Odin materials, most specific first.
DIFFUSE_SLOTS = ("diffuseTex", "diffuseTex2D", "diffuse")
EMISSION_SLOTS = ("emissionTex", "emissionTex2D")


@dataclass
class ConvertOptions:
    resolver: TextureResolver | None = None
    texture_overrides: dict[str, ResolvedTexture] = field(default_factory=dict)
    embed_textures: bool = True
    unlit: bool = False
    source_name: str | None = None


@dataclass
class AnimationInput:
    name: str
    document: GlbDocument


@dataclass
class ConvertResult:
    gltf: dict[str, Any]
    binary: bytes
    external_images: dict[str, bytes] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class _Converter:
    def __init__(self, document: GlbDocument, options: ConvertOptions):
        self.source = normalize(document.gltf)
        self.source_binary = document.binary
        self.options = options
        self.builder = BufferBuilder()
        self.warnings: list[str] = []
        self.out: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": f"bsx {__version__}"},
            "extensionsUsed": [],
            "scene": self.source.get("scene", 0),
            "scenes": self.source.get("scenes", [{"nodes": []}]),
            "nodes": [],
            "meshes": [],
            "materials": [],
            "skins": [],
            "animations": [],
            "images": [],
            "textures": [],
            "samplers": [],
            "cameras": self.source.get("cameras", []),
            "accessors": self.builder.accessors,
            "bufferViews": self.builder.buffer_views,
            "buffers": [],
        }
        if options.source_name:
            self.out["asset"]["extras"] = {"sourceFile": options.source_name}
        self.external_images: dict[str, bytes] = {}
        self._texture_cache: dict[tuple[str, bool], int] = {}
        self._odin_streams: dict[int, dict[str, np.ndarray]] = {}
        self._joint_slot_remap: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ nodes

    def convert_nodes(self) -> None:
        for node in self.source["nodes"]:
            self.out["nodes"].append({k: node[k] for k in NODE_KEYS if k in node})

    # ------------------------------------------------------------------ skins

    def convert_skins(self) -> None:
        for skin_index, skin in enumerate(self.source["skins"]):
            joints = list(skin.get("joints", []))
            matrices = None
            if skin.get("inverseBindMatrices") is not None:
                matrices = read_accessor(
                    self.source, self.source_binary, skin["inverseBindMatrices"]
                ).astype(np.float32)

            if len(set(joints)) != len(joints):
                unique: list[int] = []
                remap = np.zeros(len(joints), dtype=np.uint16)
                for slot, joint in enumerate(joints):
                    if joint not in unique:
                        unique.append(joint)
                    remap[slot] = unique.index(joint)
                self._joint_slot_remap[skin_index] = remap
                if matrices is not None:
                    keep = [joints.index(j) for j in unique]
                    matrices = matrices[keep]
                self.warnings.append(
                    f"skin {skin_index}: merged {len(joints) - len(unique)} duplicate joint(s)"
                )
                joints = unique

            out_skin: dict[str, Any] = {"joints": joints}
            if "name" in skin:
                out_skin["name"] = skin["name"]
            if skin.get("skeleton") is not None:
                out_skin["skeleton"] = skin["skeleton"]
            if matrices is not None:
                out_skin["inverseBindMatrices"] = self.builder.add_accessor(matrices)
            self.out["skins"].append(out_skin)

    # ----------------------------------------------------------------- meshes

    def _odin_attributes(self, info_index: int) -> dict[str, np.ndarray]:
        if info_index in self._odin_streams:
            return self._odin_streams[info_index]
        odin = odin_extension(self.source)
        infos = odin.get("meshDataInfos") or []
        if info_index >= len(infos) or odin.get("bufferView") is None:
            raise ValueError(f"missing Odin mesh data for meshDataInfoIndex {info_index}")
        view = buffer_view_bytes(self.source, self.source_binary, odin["bufferView"])
        info = infos[info_index]
        count = vertex_capacity(len(view), info)
        attributes = decode_mesh_data_info(view, info, count)
        self._odin_streams[info_index] = attributes
        return attributes

    def _primitive_arrays(
        self, primitive: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, np.ndarray], Any]:
        info_index = ((primitive.get("extensions") or {}).get(ODIN) or {}).get("meshDataInfoIndex")
        if info_index is not None:
            attributes = self._odin_attributes(info_index)
            key: Any = ("odin", info_index)
        else:
            attributes = {
                semantic: read_accessor(self.source, self.source_binary, accessor)
                for semantic, accessor in (primitive.get("attributes") or {}).items()
            }
            key = ("std", tuple(sorted((primitive.get("attributes") or {}).items())))

        vertex_count = min((len(a) for a in attributes.values()), default=0)
        if primitive.get("indices") is not None:
            indices = read_accessor(self.source, self.source_binary, primitive["indices"]).reshape(-1)
        else:
            indices = np.arange(vertex_count, dtype=np.uint32)
        indices = indices.astype(np.uint32)
        if len(indices) and int(indices.max()) >= vertex_count:
            raise ValueError("primitive indices exceed decoded vertex count")
        return indices, attributes, key

    @staticmethod
    def _apply_pretransform(attributes: dict[str, np.ndarray], matrix: list) -> dict[str, np.ndarray]:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 3):
            raise ValueError("inversePretransform must be a 4x3 matrix")
        linear, translation = m[:3], m[3]
        normal_matrix = np.linalg.inv(linear).T
        result = dict(attributes)
        if "POSITION" in result:
            result["POSITION"] = (result["POSITION"] @ linear + translation).astype(np.float32)
        for key in ("NORMAL", "TANGENT"):
            if key in result:
                vectors = result[key][:, :3] @ normal_matrix
                lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
                lengths[lengths == 0] = 1
                transformed = result[key].copy()
                transformed[:, :3] = (vectors / lengths).astype(np.float32)
                result[key] = transformed
        return result

    def convert_meshes(self) -> None:
        node_skin_by_mesh: dict[int, int] = {}
        for node in self.source["nodes"]:
            if node.get("mesh") is not None and node.get("skin") is not None:
                node_skin_by_mesh.setdefault(node["mesh"], node["skin"])

        for mesh_index, mesh in enumerate(self.source["meshes"]):
            odin_mesh = (mesh.get("extensions") or {}).get(ODIN) or {}
            pretransform = odin_mesh.get("inversePretransform")
            joint_remap = self._joint_slot_remap.get(node_skin_by_mesh.get(mesh_index, -1))

            decoded = [self._primitive_arrays(p) for p in mesh.get("primitives", [])]

            # Primitives that share a vertex source are compacted together so the
            # mesh only carries vertices it actually references.
            groups: dict[Any, list[int]] = {}
            for i, (_, _, key) in enumerate(decoded):
                groups.setdefault(key, []).append(i)

            accessor_sets: dict[Any, tuple[dict[str, int], np.ndarray]] = {}
            for key, members in groups.items():
                attributes = decoded[members[0]][1]
                if pretransform is not None:
                    attributes = self._apply_pretransform(attributes, pretransform)
                all_indices = np.concatenate([decoded[i][0] for i in members])
                used, inverse = np.unique(all_indices, return_inverse=True)
                remap = np.empty(int(used.max()) + 1 if len(used) else 0, dtype=np.uint32)
                remap[used] = np.arange(len(used), dtype=np.uint32)

                accessors: dict[str, int] = {}
                for semantic, array in attributes.items():
                    compact = np.ascontiguousarray(array[used])
                    if semantic == "JOINTS_0" and joint_remap is not None:
                        compact = joint_remap[compact].astype(compact.dtype)
                    accessors[semantic] = self.builder.add_accessor(
                        compact, target=ARRAY_BUFFER, min_max=(semantic == "POSITION")
                    )
                accessor_sets[key] = (accessors, remap)

            out_primitives = []
            for (indices, _, key), primitive in zip(decoded, mesh.get("primitives", [])):
                accessors, remap = accessor_sets[key]
                remapped = remap[indices]
                dtype = np.uint16 if remapped.size == 0 or remapped.max() < 65535 else np.uint32
                out_primitive: dict[str, Any] = {
                    "attributes": accessors,
                    "indices": self.builder.add_accessor(
                        remapped.astype(dtype), target=ELEMENT_ARRAY_BUFFER
                    ),
                    "mode": primitive.get("mode", 4),
                }
                if primitive.get("material") is not None:
                    out_primitive["material"] = primitive["material"]
                out_primitives.append(out_primitive)

            out_mesh: dict[str, Any] = {"primitives": out_primitives}
            if "name" in mesh:
                out_mesh["name"] = mesh["name"]
            extras = dict(mesh.get("extras") or {})
            if "bounds" in odin_mesh:
                extras["bounds"] = odin_mesh["bounds"]
            if extras:
                out_mesh["extras"] = extras
            self.out["meshes"].append(out_mesh)

    # -------------------------------------------------------------- materials

    def _texture_index(self, texture: ResolvedTexture) -> int:
        key = (texture.name, texture.repeat)
        if key in self._texture_cache:
            return self._texture_cache[key]

        image_name = Path(texture.name).stem + ".png"
        image: dict[str, Any] = {"name": Path(texture.name).stem, "mimeType": "image/png"}
        if self.options.embed_textures:
            image["bufferView"] = self.builder.add_buffer_view(texture.png)
        else:
            counter = 1
            unique_name = image_name
            while unique_name in self.external_images:
                counter += 1
                unique_name = f"{Path(texture.name).stem}_{counter}.png"
            self.external_images[unique_name] = texture.png
            image["uri"] = unique_name
        self.out["images"].append(image)

        wrap = 10497 if texture.repeat else 33071
        sampler = {"magFilter": 9729, "minFilter": 9987, "wrapS": wrap, "wrapT": wrap}
        if sampler not in self.out["samplers"]:
            self.out["samplers"].append(sampler)
        self.out["textures"].append(
            {"sampler": self.out["samplers"].index(sampler), "source": len(self.out["images"]) - 1}
        )
        index = len(self.out["textures"]) - 1
        self._texture_cache[key] = index
        return index

    def _resolve_slot(self, textures: dict[str, Any], slots: tuple[str, ...]) -> ResolvedTexture | None:
        for slot in slots:
            override = self.options.texture_overrides.get(slot)
            if override is not None:
                return override
        for slot in slots:
            reference = textures.get(slot)
            if isinstance(reference, str) and reference and self.options.resolver is not None:
                resolved = self.options.resolver.resolve(reference)
                if resolved is not None:
                    return resolved
        return None

    @staticmethod
    def _color(vectors: dict[str, Any], key: str, size: int, default: list[float]) -> list[float]:
        value = vectors.get(key)
        if not isinstance(value, list) or not value:
            return default
        values = [float(v) for v in value[:size]]
        while len(values) < size:
            values.append(1.0)
        return [min(max(v, 0.0), 1.0) for v in values]

    def convert_materials(self) -> None:
        for index, material in enumerate(self.source["materials"]):
            sc = (material.get("extensions") or {}).get(SC_SHADER)
            if sc is None:
                out = {k: v for k, v in material.items() if k != "extensions"}
                self.out["materials"].append(out)
                continue

            variables = sc.get("variables") or {}
            textures = variables.get("textures") or {}
            vectors = variables.get("floatVectors") or {}
            floats = variables.get("floats") or {}

            pbr: dict[str, Any] = {
                "baseColorFactor": self._color(vectors, "diffuse", 4, [1.0, 1.0, 1.0, 1.0]),
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            }
            out: dict[str, Any] = {
                "name": sc.get("name") or material.get("name") or f"material_{index}",
                "pbrMetallicRoughness": pbr,
                "doubleSided": bool(sc.get("doubleSided", False)),
                "extras": {SC_SHADER: sc},
            }

            diffuse = self._resolve_slot(textures, DIFFUSE_SLOTS)
            if diffuse is not None:
                pbr["baseColorTexture"] = {"index": self._texture_index(diffuse)}

            emission = self._resolve_slot(textures, EMISSION_SLOTS)
            if emission is not None:
                out["emissiveTexture"] = {"index": self._texture_index(emission)}
                out["emissiveFactor"] = self._color(vectors, "emission", 3, [1.0, 1.0, 1.0])

            cutout = floats.get("cutout")
            blend_mode = int(sc.get("blendMode", BLEND_OPAQUE))
            if isinstance(cutout, (int, float)) and cutout > 0:
                out["alphaMode"] = "MASK"
                out["alphaCutoff"] = float(cutout)
            elif blend_mode != BLEND_OPAQUE:
                out["alphaMode"] = "BLEND"

            if self.options.unlit:
                out["extensions"] = {"KHR_materials_unlit": {}}
                if "KHR_materials_unlit" not in self.out["extensionsUsed"]:
                    self.out["extensionsUsed"].append("KHR_materials_unlit")

            self.out["materials"].append(out)

    # ------------------------------------------------------------- animations

    def _add_animation(self, name: str, tracks: list[RetargetedTrack], frame_rate: float, frame_count: int) -> None:
        if frame_count == 0 or not tracks:
            self.warnings.append(f"animation {name!r} has no frames")
            return

        times = (np.arange(frame_count, dtype=np.float32) / np.float32(frame_rate)).reshape(-1, 1)
        full_input: int | None = None
        const_input: int | None = None
        samplers: list[dict[str, Any]] = []
        channels: list[dict[str, Any]] = []

        rest_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for track in tracks:
            node = self.out["nodes"][track.node]
            if track.node not in rest_cache:
                rest_cache[track.node] = node_trs(node)
            rest = dict(zip(("translation", "rotation", "scale"), rest_cache[track.node]))

            for path in ("translation", "rotation", "scale"):
                values = getattr(track, path).astype(np.float32)
                constant = bool(np.allclose(values, values[0], atol=1e-6))
                if constant:
                    rest_value = rest[path]
                    if path == "rotation" and np.dot(values[0], rest_value) < 0:
                        rest_value = -rest_value
                    if np.allclose(values[0], rest_value, atol=1e-5):
                        continue
                    if frame_count > 1:
                        if const_input is None:
                            const_input = self.builder.add_accessor(times[[0, -1]], min_max=True)
                        output = self.builder.add_accessor(values[[0, 0]])
                        input_index = const_input
                    else:
                        if full_input is None:
                            full_input = self.builder.add_accessor(times, min_max=True)
                        output = self.builder.add_accessor(values[:1])
                        input_index = full_input
                else:
                    if full_input is None:
                        full_input = self.builder.add_accessor(times, min_max=True)
                    output = self.builder.add_accessor(values)
                    input_index = full_input

                samplers.append({"input": input_index, "output": output, "interpolation": "LINEAR"})
                channels.append(
                    {"sampler": len(samplers) - 1, "target": {"node": track.node, "path": path}}
                )

        if not channels:
            self.warnings.append(f"animation {name!r} does not move any node; skipped")
            return
        self.out["animations"].append({"name": name, "samplers": samplers, "channels": channels})

    def convert_own_animation(self, name: str) -> None:
        descriptor = find_animation_descriptor(self.source)
        if descriptor is None:
            return
        animation = read_animation(self.source, self.source_binary, descriptor)
        tracks = direct_tracks(self.source["nodes"], animation)
        self._add_animation(name, tracks, animation.frame_rate, animation.frame_count)

    def convert_standard_animations(self) -> None:
        for animation in self.source.get("animations") or []:
            if ODIN in (animation.get("extensions") or {}):
                continue
            samplers = []
            for sampler in animation.get("samplers", []):
                samplers.append(
                    {
                        "input": self.builder.add_accessor(
                            read_accessor(self.source, self.source_binary, sampler["input"]).astype(np.float32),
                            min_max=True,
                        ),
                        "output": self.builder.add_accessor(
                            read_accessor(self.source, self.source_binary, sampler["output"]).astype(np.float32)
                        ),
                        "interpolation": sampler.get("interpolation", "LINEAR"),
                    }
                )
            if samplers and animation.get("channels"):
                self.out["animations"].append(
                    {
                        "name": animation.get("name", "animation"),
                        "samplers": samplers,
                        "channels": animation["channels"],
                    }
                )

    def add_external_animation(self, clip: AnimationInput) -> None:
        source = normalize(clip.document.gltf)
        descriptor = find_animation_descriptor(source)
        if descriptor is None:
            self.warnings.append(f"{clip.name}: no Odin animation found; skipped")
            return
        animation = read_animation(source, clip.document.binary, descriptor)
        tracks, unmatched = retarget(source["nodes"], animation, self.out["nodes"])
        if unmatched:
            log.debug(
                "%s: %d animated helper node(s) without counterpart in the target rig "
                "(e.g. %s); their motion is folded into matched children",
                clip.name,
                len(unmatched),
                ", ".join(unmatched[:4]),
            )
        if not tracks:
            self.warnings.append(f"{clip.name}: no node names match the target rig; skipped")
            return
        self._add_animation(clip.name, tracks, animation.frame_rate, animation.frame_count)

    # ----------------------------------------------------------------- finish

    def finish(self) -> ConvertResult:
        binary = self.builder.finish()
        self.out["buffers"] = [{"byteLength": len(binary)}] if binary else []
        for key in ("materials", "skins", "animations", "images", "textures", "samplers", "cameras", "extensionsUsed"):
            if not self.out[key]:
                del self.out[key]
        if not self.out["accessors"]:
            del self.out["accessors"]
            del self.out["bufferViews"]
        if not self.out["buffers"]:
            del self.out["buffers"]
        if not self.out["meshes"]:
            del self.out["meshes"]
        return ConvertResult(self.out, binary, self.external_images, self.warnings)


def convert(
    document: GlbDocument,
    options: ConvertOptions | None = None,
    animations: list[AnimationInput] | None = None,
    own_animation_name: str = "animation",
) -> ConvertResult:
    """Convert an Odin (or plain) GLB document to standard glTF 2.0.

    ``animations`` are additional Odin animation files whose clips are
    retargeted by node name onto the document's node hierarchy.
    """
    options = options or ConvertOptions()
    converter = _Converter(document, options)
    converter.convert_nodes()
    converter.convert_skins()
    converter.convert_meshes()
    converter.convert_materials()
    converter.convert_standard_animations()
    converter.convert_own_animation(own_animation_name)
    for clip in animations or []:
        converter.add_external_animation(clip)
    return converter.finish()
