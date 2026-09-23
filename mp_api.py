"""Thin client for the Manifesto Project REST API, standard library only.

Every response is cached as JSON under data/raw/ so repeated runs do not
spend the daily request quota. Delete the cache to force a refetch.
The key is sent as a header so it never appears in a URL or a log.
"""

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://manifesto-project.wzb.eu/api/v1/"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "raw"
KEY_FILE = ROOT / "manifesto_apikey.txt"


def api_key():
    key = os.environ.get("MANIFESTO_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(
            "No API key. Set MANIFESTO_API_KEY or put it in manifesto_apikey.txt"
        )
    return key


def _cache_path(endpoint, params):
    blob = json.dumps([endpoint, params], sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()[:16]
    return CACHE_DIR / f"{endpoint}_{digest}.json"


def call(endpoint, params=None, use_cache=True):
    """GET an endpoint. List values are sent as repeated name[] params."""
    params = params or {}
    path = _cache_path(endpoint, params)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    query = []
    for name, value in params.items():
        if isinstance(value, (list, tuple)):
            query.extend((f"{name}[]", v) for v in value)
        else:
            query.append((name, value))
    url = BASE_URL + endpoint + "?" + urllib.parse.urlencode(query)
    req = urllib.request.Request(url, headers={"API_KEY": api_key()})

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            body = err.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{endpoint} returned {err.code}: {body}") from None
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
