"""Client for the public sc-workshop asset index (``files.sc-workshop.com``).

The service indexes the asset bundles of several Supercell games per released
version. Requests are plain GET calls; the ``request`` parameter is a space
separated list of ``regex tokens``, the game codename and an optional
``v<version>`` suffix, for example ``"shelly_geo .glb$ BS v69.270.1"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

ASSETS_URL = "https://files.sc-workshop.com/asset-request/"
SERVERS_URL = "https://files.sc-workshop.com/servers/"
DEFAULT_TIMEOUT = 30
DEFAULT_GAME = "BS"


class AssetApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class GameServer:
    name: str
    codename: str


@dataclass(frozen=True)
class GameVersion:
    version: str
    hash: str


def _request_string(tokens: list[str], game: str, version: str | None) -> str:
    parts = ["|".join(t for t in tokens if t), game]
    if version:
        parts.append(f"v{version}")
    return " ".join(parts)


class AssetClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise AssetApiError(f"request to {url} failed: {exc}") from exc
        if response.status_code != 200:
            raise AssetApiError(
                f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
            )
        return response

    def servers(self) -> list[GameServer]:
        return [GameServer(**item) for item in self._get(SERVERS_URL).json()]

    def versions(self, game: str = DEFAULT_GAME) -> list[GameVersion]:
        params = {
            "type": "listVersions",
            "request": _request_string([], game, None),
            "compressed": "false",
        }
        return [GameVersion(**item) for item in self._get(ASSETS_URL, params).json()]

    def search(
        self,
        pattern: str,
        game: str = DEFAULT_GAME,
        version: str | None = None,
        limit: int = 50,
        page: int = 0,
    ) -> list[str]:
        """List asset paths matching ``pattern`` (regex tokens separated by spaces)."""
        params = {
            "type": "listAssets",
            "request": _request_string(pattern.split(), game, version),
            "compressed": "false",
            "size": limit,
            "index": page,
        }
        result = self._get(ASSETS_URL, params).json()
        return [str(item) for item in result] if isinstance(result, list) else []

    def download(self, path: str, game: str = DEFAULT_GAME, version: str | None = None) -> bytes:
        """Download one asset by its exact path (e.g. ``sc3d/shelly_geo.glb``)."""
        params = {
            "type": "downloadAssets",
            "request": _request_string([path], game, version),
            "compressed": "false",
            "size": 1,
            "index": 0,
        }
        response = self._get(ASSETS_URL, params)
        content = response.content
        if not content:
            raise AssetApiError(f"asset {path!r} not found")
        return content


def cache_dir() -> Path:
    root = os.environ.get("BSX_CACHE_DIR")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "bsx"


def fetch_cached(
    client: AssetClient, path: str, game: str = DEFAULT_GAME, version: str | None = None
) -> Path:
    """Download an asset into the local cache (if needed) and return its path."""
    target = cache_dir() / game / (version or "latest") / path
    if target.is_file() and target.stat().st_size > 0:
        return target
    data = client.download(path, game, version)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target
