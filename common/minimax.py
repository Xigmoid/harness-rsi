"""The MiniMax side-car: a second model provider for the annotation run.

Why
---

The annotation runs against **one** GLM endpoint, and that endpoint is
shared and heavily loaded.  When it is slow, a trial does not fail: it
*stalls*, and a stalled trial hits the agent timeout and dies with nothing
to show for the minutes it spent waiting.  A second provider turns that
from a loss into a reroute.

Two things are done with it, and they are deliberately different:

**As a fallback.**  A GLM request that fails -- a 5xx, a closed
connection, a rate limit -- is retried against MiniMax instead of killing
the rollout.  The rollout continues, and the rows that used it are marked.

**As a share of the work.**  ``share_of()`` decides, per task and
deterministically, whether the task is annotated by GLM or by MiniMax.
The decision is a hash of the task name, so it is the same on every
re-run, it does not depend on the row order, and it is the *same* rule the
harness split uses -- so the two splits cannot collide.

The provenance is never lost: every dataset row carries the model that
produced it, so a GLM-only dataset can be selected out of a mixed one
without re-running anything.

The API
-------

``https://api.minimax.cn/v1`` with the model ``MiniMax-M3.1-Flash``, an
OpenAI-compatible Chat Completions endpoint.  The wire shape is the one the
harnesses already speak: ``tool_calls[].function.arguments`` is a JSON
*string*, the reasoning comes back in ``reasoning_content``, and the model
id comes back in ``model``.  Nothing had to be adapted.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

#: The OpenAI-compatible endpoint.
MINIMAX_BASE = "https://api.minimax.cn/v1"

#: The model id.
MINIMAX_MODEL = "MiniMax-M3.1-Flash"

#: The litellm model name the mini harness uses for it.  mini goes through
#: litellm's ``openai`` provider, which is the same provider the GLM
#: endpoint uses -- only the base url and the key differ.
MINIMAX_LITELLM_NAME = f"openai/{MINIMAX_MODEL}"

#: Where the key lives, relative to the repository root.
DEFAULT_KEY_FILE = "MINIMAX_KEY.txt"


def read_key(path: str | Path = DEFAULT_KEY_FILE) -> str:
    """The API key, read from the file that holds it.

    The key is never written into a log, a job file or a dataset row: the
    job file goes into the container, and the containers are ephemeral but
    the files are not.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    return text


def share_of(name: str, share: float) -> bool:
    """Does this task belong to the MiniMax share?

    The rule is the same one the harness split uses -- the sha256 of the
    task name, first four bytes, modulo -- so the two decisions are
    independent and a task cannot end up on both sides.
    """
    if share <= 0:
        return False
    if share >= 1:
        return True
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    position = int.from_bytes(digest[:4], "big") / 2**32
    return position < share


def chat(
    base: str,
    key: str,
    messages: list[dict[str, Any]],
    model: str = MINIMAX_MODEL,
    timeout: float = 300.0,
    **extra: Any,
) -> dict[str, Any]:
    """One Chat Completions call, without any SDK in the way.

    The raw HTTP call is deliberate: the pipeline must work in the
    container, where there is no MiniMax SDK, and the request body has to
    be exactly the one the harnesses send.
    """
    body = json.dumps(
        {"model": model, "messages": messages, **extra}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def probe(base: str = MINIMAX_BASE, key: str | None = None) -> dict[str, Any]:
    """Is the endpoint answering, and does it do tool calls?

    The tool-call part matters more than the hello: an agent harness that
    cannot ask for a tool is useless, and a provider that answers a chat
    but not a tool call would fail in the middle of a rollout instead of
    here.
    """
    if key is None:
        key = read_key()
    response = chat(
        base,
        key,
        [{"role": "user", "content": "Call the ping tool."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        max_completion_tokens=2000,
    )
    message = (response.get("choices") or [{}])[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    return {
        "base": base,
        "model": response.get("model"),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "n_tool_calls": len(tool_calls),
        "tool_name": (tool_calls[0].get("function") or {}).get("name") if tool_calls else None,
        "usage": response.get("usage"),
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(probe(), indent=2, ensure_ascii=False))
    sys.exit(0)
