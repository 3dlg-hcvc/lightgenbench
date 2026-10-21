"""What the category steps share: the paths and model names, the VLM output schemas, the
resolver from split ids to uuids, the OpenAI client (sync and async) and the subcommand runner.

The paths other stages read too come from the repo's `utils/`. `openai` and `pandas` load where
they are used, so the steps that call no API (`composites`, `cluster`, `relabel`) import without
the OpenAI package.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from utils.paths import (
    CATEGORIES_OUT_FULL, OPENAI_KEY_PATH, TEXVERSE_METADATA_PARQUET, TEXVERSE_THUMBNAIL_ROOT,
)
from utils.review_images import LEGACY_ORBIT_ROOT

if TYPE_CHECKING:
    import pandas as pd
    from openai import AsyncOpenAI, OpenAI


# ---------------------------------------------------------------------------------------------
# Paths, model names and sampling parameters

#: uuid -> `ditem_dir`, whose leading component is the multiview tree's bucket.
METADATA_PARQUET = TEXVERSE_METADATA_PARQUET
THUMBNAILS_ROOT = TEXVERSE_THUMBNAIL_ROOT
#: The category run's orbit views: the legacy orbit root of the review images.
MULTIVIEW_ROOT = LEGACY_ORBIT_ROOT

# OpenAI API key file; falls back to the OPENAI_API_KEY env var if this file doesn't exist.
KEY_CONFIG_PATH = OPENAI_KEY_PATH

#: Every record of the category run.
OUT_FULL_DIR = CATEGORIES_OUT_FULL
COMPOSITES_FULL = OUT_FULL_DIR / "composites_2x2"

VLM_MODEL = "gpt-4o-mini"
TEXT_EMBED_MODEL = "text-embedding-3-small"

SAMPLING_SEED = 42

PANE_SIZE = 512

MULTIVIEW_AXES = ("+x_to_-x", "-y_to_+y", "+z_to_-z")


# ---------------------------------------------------------------------------------------------
# Pydantic schemas of the VLM answers

class Pass1Output(BaseModel):
    """Free-form labeling output (per object)."""
    model_config = ConfigDict(extra="forbid")
    category_freeform: str = Field(min_length=1, max_length=80)


class ClusterNamingOutput(BaseModel):
    """Cluster-naming VLM output."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    definition: str = Field(min_length=1, max_length=200)


# ---------------------------------------------------------------------------------------------
# uuid -> bucket
#
# The bucket is the leading component of the TexVerse metadata's `ditem_dir` (e.g. "000-000"),
# the <bucket> directory of the multiview tree; the uuid names the thumbnail and the multiview
# <id> directory.

@dataclass(frozen=True)
class ResolvedId:
    uuid: str
    bucket: str


def resolve_buckets(uuids: List[str], metadata_parquet: Path) -> List[ResolvedId]:
    import pandas as pd

    dirs = pd.read_parquet(metadata_parquet, columns=["ditem_dir"])["ditem_dir"]
    dirs = dirs[~dirs.index.duplicated(keep="first")]
    missing = [u for u in uuids if u not in dirs.index]
    if missing:
        raise KeyError(f"{len(missing)} uuids not in {metadata_parquet}, e.g. {missing[:3]}")
    return [ResolvedId(uuid=u, bucket=str(dirs[u]).split("/", 1)[0]) for u in uuids]


# ---------------------------------------------------------------------------------------------
# The OpenAI client
#
# Chat Completions with structured outputs (response_format = json_schema), local images sent
# as base64 data URLs (detail: low), exponential backoff on any error. `VlmClient` makes one
# call at a time (the cluster naming and the taxonomy merges); `AsyncVlmClient` keeps many in
# flight under the caller's semaphore (the two classify passes and the backfill) and adds
# jitter to its backoff so concurrent retries do not synchronize.

T = TypeVar("T", bound=BaseModel)


def _encode_image_data_url(path: Path) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def make_openai_client() -> OpenAI:
    """Build an OpenAI client from the key file `KEY_CONFIG_PATH`, else the OPENAI_API_KEY env var."""
    from openai import OpenAI

    if KEY_CONFIG_PATH.exists():
        api_key = KEY_CONFIG_PATH.read_text().strip()
        return OpenAI(api_key=api_key)
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI()
    raise SystemExit(
        f"No API key found: tried {KEY_CONFIG_PATH} and OPENAI_API_KEY env var."
    )


def make_async_openai_client() -> AsyncOpenAI:
    """Mirror of make_openai_client but for AsyncOpenAI."""
    from openai import AsyncOpenAI

    if KEY_CONFIG_PATH.exists():
        api_key = KEY_CONFIG_PATH.read_text().strip()
        return AsyncOpenAI(api_key=api_key)
    if os.environ.get("OPENAI_API_KEY"):
        return AsyncOpenAI()
    raise SystemExit(
        f"No API key found: tried {KEY_CONFIG_PATH} and OPENAI_API_KEY env var."
    )


class VlmClient:
    def __init__(self, client: Optional[OpenAI] = None, model: str = "gpt-4o-mini",
                 max_retries: int = 4, backoff_base: float = 1.0):
        from openai import OpenAI

        self._client = client if client is not None else OpenAI()
        self._model = model
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    def call_structured(
        self,
        prompt: str,
        image_paths: Iterable[Path],
        schema: Type[T],
        detail: str = "low",
        temperature: float = 0.0,
    ) -> T:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image_data_url(Path(p)), "detail": detail},
            })

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": content}],
                    response_format=response_format,
                    temperature=temperature,
                )
                payload = resp.choices[0].message.content
                return schema.model_validate(json.loads(payload))
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_base * (2 ** attempt))
        raise RuntimeError(f"VLM call failed after {self._max_retries} attempts: {last_err}")


class AsyncVlmClient:
    """Async gpt-4o-mini wrapper with retry."""

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini",
                  max_retries: int = 4, backoff_base: float = 1.0):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    async def call_structured(
        self,
        prompt: str,
        image_paths: Iterable[Path],
        schema: Type[T],
        detail: str = "low",
        temperature: float = 0.0,
    ) -> T:
        content: List[dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image_data_url(Path(p)), "detail": detail},
            })

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": content}],
                    response_format=response_format,
                    temperature=temperature,
                )
                payload = resp.choices[0].message.content
                return schema.model_validate(json.loads(payload))
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff with jitter so concurrent retries
                    # don't synchronize.
                    sleep_s = self._backoff_base * (2 ** attempt) * (1 + random.random() * 0.5)
                    await asyncio.sleep(sleep_s)
        raise RuntimeError(f"VLM call failed after {self._max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------------------------
# The subcommand runner of `taxonomy.py` and `classify.py`

def run_subcommand(commands, doc, argv=None, prog=None):
    """`<module> <command> [flags]` runs `commands[command](flags)`. Each command parses its own
    flags, so `<module> <command> -h` lists them and `<module> -h` lists the commands."""
    ap = argparse.ArgumentParser(prog=prog, description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in commands:
        sub.add_parser(name, add_help=False, help=f"`{name} -h` lists its flags")
    a, rest = ap.parse_known_args(argv)
    return commands[a.cmd](rest)
