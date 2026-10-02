"""NVIDIA Kumo Relational (formerly KumoRFM) on the hosted NVIDIA API Catalog.

KumoRFM moved under NVIDIA in 2026 and is now served as the ``kumo-relational``
model at ``https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational``.
The catalog authenticates with ``Authorization: Bearer nvapi-...``.

NVIDIA's own Python client (``kumo-relational-client`` + ``kumo-relational-engine``)
does the hard part: it turns a pandas graph and a PQL query into the
Universal Structured Data request (per-entity subgraphs plus in-context labeled
examples). Its HTTP layer, though, targets a self-hosted NIM: it sends the key
as ``X-API-Key`` and posts to ``<url>/v1/predictions``. The hosted catalog wants
Bearer auth and ``<url>/predictions``. :class:`CatalogClient` subclasses the
engine's ``NimClient`` to fix only those two things, so every request the
engine builds goes to the catalog unchanged.

The key is read from ``KUMO_API_KEY`` (or ``NVIDIA_API_KEY``), loading ``.env``
from the working directory or the repo root first. The key is never logged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CATALOG_URL = "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational"
KEY_ENVS = ("KUMO_API_KEY", "NVIDIA_API_KEY")
URL_ENV = "KUMO_RELATIONAL_CATALOG_URL"
SIGNUP_URL = "https://build.nvidia.com/nvidia/kumo-relational"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load ``.env`` from the cwd and the repo root. Existing env vars win."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional at import time; requirements include it
        return
    for path in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)


def api_key() -> str | None:
    load_env()
    for name in KEY_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def is_nvidia_key(key: str | None) -> bool:
    return bool(key) and key.startswith("nvapi-")


def redact(text: str, key: str | None = None) -> str:
    key = key or api_key()
    if key:
        text = text.replace(key, "<redacted>")
    return text


def _nim_client_cls():
    from kumo_relational_engine.client.client import NimClient

    class CatalogClient(NimClient):
        """``NimClient`` with Bearer auth and the catalog's path layout."""

        def __init__(self, key: str, url: str, timeout: float = 300.0, max_retries: int = 2):
            # api_key=None: the parent would set X-API-Key. We set Bearer instead.
            super().__init__(url, api_key=None, timeout=timeout, max_retries=max_retries)
            self._session.headers["Authorization"] = f"Bearer {key}"
            self._session.headers["Accept"] = "application/json"

        def _format_endpoint_url(self, endpoint: str) -> str:
            # /v1/predictions -> <catalog>/predictions
            if endpoint.startswith("/v1/"):
                endpoint = endpoint[3:]
            return f"{self._url}{endpoint}"

        def authenticate(self) -> None:  # the catalog has no /v1/health/ready
            return None

    return CatalogClient


def catalog_client(key: str | None = None, *, timeout: float = 300.0):
    key = key or api_key()
    if not key:
        raise RuntimeError(
            f"KUMO_API_KEY is not set. Create an NVIDIA API key at {SIGNUP_URL} "
            "and put KUMO_API_KEY=nvapi-... in .env"
        )
    # The catalog is stateless. Skip the engine's /v1/sessions attempt.
    os.environ.setdefault("KUMO_RELATIONAL_DISABLE_SESSIONS", "1")
    url = os.environ.get(URL_ENV, CATALOG_URL).rstrip("/")
    return _nim_client_cls()(key, url, timeout=timeout)


@dataclass
class Prediction:
    query: str
    frame: pd.DataFrame
    seconds: float


class KumoRelational:
    """A graph bound to the hosted model. ``predict`` takes PQL, returns a DataFrame."""

    def __init__(self, tables: dict[str, pd.DataFrame], *, key: str | None = None, verbose: bool = False, **graph_kwargs: Any):
        from kumo_relational_engine.rfm import Graph
        from kumo_relational_engine.rfm import KumoRelational as _Engine

        self.graph = Graph.from_data(tables, verbose=verbose, **graph_kwargs)
        self._engine = _Engine(self.graph, verbose=verbose, _client=catalog_client(key))

    def predict(self, query: str, **kwargs: Any) -> Prediction:
        kwargs.setdefault("verbose", False)
        start = time.perf_counter()
        try:
            # batch_mode('max') splits large entity lists into the largest
            # batch the task allows (e.g. 200 for ranking); small lists are one call.
            with self._engine.batch_mode(batch_size="max", num_retries=1):
                frame = self._engine.predict(query, **kwargs)
        except Exception as exc:  # re-raise without the key anywhere in the text
            raise RuntimeError(redact(f"{type(exc).__name__}: {exc}")) from None
        seconds = time.perf_counter() - start
        if not isinstance(frame, pd.DataFrame) and hasattr(frame, "prediction"):
            frame = frame.prediction
        return Prediction(query=query, frame=frame, seconds=seconds)


_TRUE = {"true", "1", "1.0", "yes"}
_FALSE = {"false", "0", "0.0", "no"}


def positive_scores(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Collapse a ``CLASS``/``SCORE`` frame with boolean classes to one ``TRUE_PROB`` per entity.

    Returns ``None`` when the classes are not boolean (e.g. a ranking).
    """
    if not {"ENTITY", "CLASS", "SCORE"} <= set(frame.columns):
        return None
    labels = frame["CLASS"].astype(str).str.lower()
    if not labels.isin(_TRUE | _FALSE).all():
        return None
    rows = []
    for entity, group in frame.assign(_label=labels).groupby("ENTITY", sort=False):
        pos = group.loc[group["_label"].isin(_TRUE), "SCORE"]
        if len(pos):
            p = float(pos.max())
        else:
            p = 1.0 - float(group["SCORE"].max())
        rows.append((entity, p))
    return pd.DataFrame(rows, columns=["ENTITY", "TRUE_PROB"])


def top_class(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Top-scored ``CLASS`` per entity for ranking / multiclass frames."""
    if not {"ENTITY", "CLASS", "SCORE"} <= set(frame.columns):
        return None
    best = frame.sort_values("SCORE", ascending=False).drop_duplicates("ENTITY")
    return best[["ENTITY", "CLASS", "SCORE"]].reset_index(drop=True)


def predict(tables: dict[str, pd.DataFrame], query: str, **kwargs: Any) -> pd.DataFrame:
    """One-shot helper: build the graph, run one PQL query."""
    return KumoRelational(tables).predict(query, **kwargs).frame
