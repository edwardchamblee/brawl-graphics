"""Transfer Odin animation tracks from one node hierarchy onto another.

Brawl Stars ships geometry and animations in separate files with different
rigs: the animation rig contains helper nodes (``hips_s:SSC``, ``torso``...)
that the geometry rig does not have, and those helpers are often the ones
carrying the translation. Copying local transforms by name would therefore be
wrong. Instead the animation rig is evaluated in world space per frame and
every geometry node that has a namesake receives the local transform that
reproduces that world transform under its own parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..odin.animation import NodeTrack, OdinAnimation
from .transform import (
    compose,
    decompose,
    ensure_shortest_path,
    node_local_matrix,
    node_trs,
    parent_map,
    topological_order,
)


@dataclass
class RetargetedTrack:
    node: int
    translation: np.ndarray  # (F, 3)
    rotation: np.ndarray  # (F, 4)
    scale: np.ndarray  # (F, 3)


def _track_value(track: NodeTrack, path: str, frame: int) -> np.ndarray:
    values = track.channel(path)
    if len(values) == 0:
        return {
            "translation": np.zeros(3),
            "rotation": np.array([0, 0, 0, 1.0]),
            "scale": np.ones(3),
        }[path]
    return values[min(frame, len(values) - 1)]


def world_matrices(
    nodes: list[dict[str, Any]],
    tracks: dict[int, NodeTrack],
    frame: int,
    order: list[int] | None = None,
    parents: dict[int, int | None] | None = None,
) -> list[np.ndarray]:
    """World matrix of every node for one animation frame."""
    order = order or topological_order(nodes)
    parents = parents or parent_map(nodes)
    world: list[np.ndarray] = [np.eye(4)] * len(nodes)
    for index in order:
        track = tracks.get(index)
        if track is None:
            local = node_local_matrix(nodes[index])
        else:
            local = compose(
                _track_value(track, "translation", frame),
                _track_value(track, "rotation", frame),
                _track_value(track, "scale", frame),
            )
        parent = parents.get(index)
        world[index] = local if parent is None else world[parent] @ local
    return world


def retarget(
    source_nodes: list[dict[str, Any]],
    animation: OdinAnimation,
    target_nodes: list[dict[str, Any]],
) -> tuple[list[RetargetedTrack], list[str]]:
    """Return per-frame TRS for every target node that exists in the source rig.

    Also returns the names of source nodes that had no counterpart.
    """
    frame_count = animation.frame_count
    source_tracks = {track.node: track for track in animation.tracks}

    source_by_name: dict[str, int] = {}
    for index, node in enumerate(source_nodes):
        name = node.get("name")
        if name is not None and name not in source_by_name:
            source_by_name[name] = index

    matches: dict[int, int] = {}
    for index, node in enumerate(target_nodes):
        source_index = source_by_name.get(node.get("name"))
        if source_index is not None:
            matches[index] = source_index

    animated_names = {source_nodes[t.node].get("name") for t in animation.tracks}
    matched_names = {source_nodes[s].get("name") for s in matches.values()}
    unmatched = sorted(str(n) for n in animated_names - matched_names if n is not None)

    source_order = topological_order(source_nodes)
    source_parents = parent_map(source_nodes)
    target_order = topological_order(target_nodes)
    target_parents = parent_map(target_nodes)
    target_rest = [node_local_matrix(node) for node in target_nodes]

    results: dict[int, RetargetedTrack] = {
        index: RetargetedTrack(
            node=index,
            translation=np.zeros((frame_count, 3), dtype=np.float32),
            rotation=np.zeros((frame_count, 4), dtype=np.float32),
            scale=np.ones((frame_count, 3), dtype=np.float32),
        )
        for index in matches
    }

    for frame in range(frame_count):
        source_world = world_matrices(
            source_nodes, source_tracks, frame, source_order, source_parents
        )
        target_world: list[np.ndarray] = [np.eye(4)] * len(target_nodes)
        for index in target_order:
            parent = target_parents.get(index)
            parent_world = np.eye(4) if parent is None else target_world[parent]
            source_index = matches.get(index)
            if source_index is None:
                local = target_rest[index]
            else:
                world = source_world[source_index]
                local = np.linalg.inv(parent_world) @ world
                t, r, s = decompose(local)
                track = results[index]
                track.translation[frame] = t
                track.rotation[frame] = r
                track.scale[frame] = s
            target_world[index] = parent_world @ local

    for track in results.values():
        track.rotation = ensure_shortest_path(track.rotation)

    return list(results.values()), unmatched


def _track_channel(
    track: NodeTrack, path: str, rest: np.ndarray, width: int, frame_count: int
) -> np.ndarray:
    if not track.animated.get(path):
        return np.tile(rest.astype(np.float32), (frame_count, 1))
    values = track.channel(path)
    if len(values) < frame_count:
        pad = np.repeat(values[-1:], frame_count - len(values), axis=0)
        values = np.concatenate([values, pad])
    return values[:frame_count].astype(np.float32).reshape(frame_count, width)


def direct_tracks(nodes: list[dict[str, Any]], animation: OdinAnimation) -> list[RetargetedTrack]:
    """Use decoded tracks as-is on the rig they were authored for."""
    frame_count = animation.frame_count
    results: list[RetargetedTrack] = []
    for track in animation.tracks:
        if not (0 <= track.node < len(nodes)):
            continue
        rest_t, rest_r, rest_s = node_trs(nodes[track.node])
        results.append(
            RetargetedTrack(
                node=track.node,
                translation=_track_channel(track, "translation", rest_t, 3, frame_count),
                rotation=ensure_shortest_path(
                    _track_channel(track, "rotation", rest_r, 4, frame_count)
                ),
                scale=_track_channel(track, "scale", rest_s, 3, frame_count),
            )
        )
    return results
