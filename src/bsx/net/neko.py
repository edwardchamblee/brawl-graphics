"""Fallback texture conversion through the sc-workshop "Neko" web API.

Local decoding (``bsx.textures.sctx``) handles the formats seen in Brawl Stars.
This client exists for pixel formats the local decoder does not implement:
``POST /v1/texture/png`` converts an uploaded ``.sctx`` and returns two hex
handles, which ``POST /v1/cdn`` exchanges for the PNG bytes.
"""

from __future__ import annotations

import requests

NEKO_URL = "https://api.sc-workshop.com"
DEFAULT_TIMEOUT = 60


class NekoApiError(RuntimeError):
    pass


def convert_sctx_to_png(
    name: str, data: bytes, session: requests.Session | None = None, timeout: float = DEFAULT_TIMEOUT
) -> bytes:
    session = session or requests.Session()
    try:
        response = session.post(
            f"{NEKO_URL}/v1/texture/png",
            params={"requestType": "user", "source": 0},
            files={"files": (name, data)},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise NekoApiError(f"texture conversion request failed: {exc}") from exc
    if response.status_code != 200:
        raise NekoApiError(f"texture conversion failed (HTTP {response.status_code}): {response.text[:200]}")

    reference = response.json()
    try:
        cdn = session.post(
            f"{NEKO_URL}/v1/cdn",
            params={"high": reference["high"], "low": reference["low"]},
            timeout=timeout,
        )
    except (requests.RequestException, KeyError) as exc:
        raise NekoApiError(f"texture download failed: {exc}") from exc
    if cdn.status_code != 200:
        raise NekoApiError(f"texture download failed (HTTP {cdn.status_code})")
    return cdn.content
