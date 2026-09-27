"""Small TRS / matrix helpers (glTF conventions: column vectors, xyzw quaternions)."""

from __future__ import annotations

from typing import Any

import numpy as np


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3, orthonormal) -> unit quaternion xyzw."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def compose(t: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quat_to_matrix(r) * np.asarray(s, dtype=np.float64)[None, :]
    m[:3, 3] = t
    return m


def decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4x4 affine matrix -> (translation, quaternion xyzw, scale)."""
    t = m[:3, 3].copy()
    linear = m[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.linalg.det(linear) < 0:
        scale = -scale
    safe = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    rot = linear / safe[None, :]
    # Re-orthonormalise to keep the quaternion well-defined for slightly sheared inputs.
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return t, matrix_to_quat(rot), scale


def node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    return compose(t, r, s)


def node_trs(node: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "matrix" in node:
        return decompose(node_local_matrix(node))
    return (
        np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
        np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64),
        np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64),
    )


def parent_map(nodes: list[dict[str, Any]]) -> dict[int, int | None]:
    parents: dict[int, int | None] = {i: None for i in range(len(nodes))}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = index
    return parents


def topological_order(nodes: list[dict[str, Any]]) -> list[int]:
    """Node indices ordered so that every parent precedes its children."""
    parents = parent_map(nodes)
    order: list[int] = []
    visited: set[int] = set()

    def visit(index: int) -> None:
        if index in visited:
            return
        parent = parents.get(index)
        if parent is not None:
            visit(parent)
        visited.add(index)
        order.append(index)

    for index in range(len(nodes)):
        visit(index)
    return order


def ensure_shortest_path(quats: np.ndarray) -> np.ndarray:
    """Flip antipodal quaternions so consecutive keys interpolate the short way."""
    out = quats.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out
