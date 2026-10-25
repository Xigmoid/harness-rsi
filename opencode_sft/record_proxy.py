"""A recording proxy in front of an OpenAI-compatible endpoint.

opencode's system prompt and its tool schemas are baked into the binary and
are **not** exposed by ``opencode debug agent`` -- the agent dump only has the
permissions and the tool list.  The only faithful way to capture what the
model actually saw is to record the request.

This proxy sits between opencode and the real endpoint, forwards every
request unchanged, and appends each request body to a JSONL file.  From one
recorded request you get:

* the exact system prompt (and the ``AGENTS.md`` additions, if any);
* the exact tool schemas the model was offered;
* the exact message list the model was given, i.e. the real LLM context --
  which is what an SFT dataset has to reproduce.

Usage::

    python opencode_sft/record_proxy.py --port 8899 \
        --upstream http://YOUR-ENDPOINT:PORT/v1 \
        --out /tmp/opencode-requests.jsonl

then point opencode at ``http://127.0.0.1:8899/v1``.

The proxy is a pass-through: it never rewrites the body, and it streams the
response back so that long rollouts are not buffered.

Scaling to 31,600 rollouts
--------------------------

A full request body is ~82 KB on average and grows with the conversation, so
recording every request of every rollout would write tens of gigabytes.  But
the converter only ever reads three things out of a request: the leading
``system`` message, the first ``user`` message, and the tool schemas.  So
``--compact`` keeps exactly those and drops the rest of the conversation,
and it writes **one record per rollout**: opencode's first user message
identifies the rollout, and every later request of the same rollout carries
the same system prompt and the same tools.

The dedup key is the pair *(first user message, system prompt digest)*, not
the user message alone, so a task that is rolled out twice on two different
days -- the system prompt carries the date -- still gets a record per day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencode_sft.minimax import read_key  # noqa: E402

#: Request bodies larger than this are not recorded (they would be a runaway
#: context, not a rollout).
MAX_RECORD_BYTES = 8 * 1024 * 1024

#: Headers that must not be copied to the upstream request.
SKIP_HEADERS = {"host", "content-length", "connection", "accept-encoding"}


def _first_user_message(body: dict[str, Any]) -> str | None:
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) and content else None
    return None


def _system_digest(body: dict[str, Any]) -> str:
    parts = []
    for message in body.get("messages") or []:
        if message.get("role") == "system":
            content = message.get("content")
            parts.append(
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, sort_keys=True)
            )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def compact_body(body: dict[str, Any]) -> dict[str, Any]:
    """The part of a request the converter actually reads.

    The leading ``system`` messages and the first ``user`` message are kept,
    in their original order; the rest of the conversation -- which the
    converter rebuilds from opencode's own session export -- is dropped.
    """
    kept: list[dict[str, Any]] = []
    user_kept = False
    for message in body.get("messages") or []:
        role = message.get("role")
        if role == "system":
            kept.append(message)
        elif role == "user" and not user_kept:
            kept.append(message)
            user_kept = True
    return {
        "model": body.get("model"),
        "messages": kept,
        "tools": body.get("tools"),
    }


class BlobStore:
    """Stores the two parts of a request that every rollout repeats.

    The opencode system prompt is ~80 KB and the tool schemas are ~10 KB, and
    the converter needs **both for every row** -- they are not in the session
    export.  Written out per rollout they cost ~90 KB per row, which is 700 MB
    for the 7,653 rollouts of the ClawGym opencode half.

    Inside a container the system prompt is the *same* for every rollout of
    the same day: the same working directory, the same skills on disk, the
    same date.  And the tool schemas are the same for every rollout at all.
    So the two are written **once**, by content digest, and the request log
    only carries the digests.  The cost per rollout drops from ~90 KB to
    ~120 bytes, and the log stays readable.

    The blobs are content-addressed, so a second run that produces the same
    system prompt reuses the file that is already there.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.n_blobs = 0

    def _write(self, kind: str, payload: Any) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        path = self.directory / f"{digest}.{kind}.json"
        with self._lock:
            if not path.exists():
                # The write is atomic: a reader that finds the file finds a
                # whole blob, never a half-written one.
                tmp = path.with_name(path.name + ".tmp")
                with tmp.open("w", encoding="utf-8") as handle:
                    handle.write(blob)
                os.replace(tmp, path)
                self.n_blobs += 1
        return digest

    def put_system(self, messages: list[dict[str, Any]]) -> str:
        return self._write("system", {"messages": messages})

    def put_tools(self, tools: Any) -> str:
        return self._write("tools", tools)


def compact_body_with_blobs(
    body: dict[str, Any], blobs: BlobStore
) -> dict[str, Any]:
    """The compact body, with the system prompt and the tools behind digests.

    The shape is the same as ``compact_body()``'s -- a ``messages`` list and a
    ``tools`` list -- except that the repeated parts are replaced by the
    digest of the blob that holds them.
    """
    system_messages = [
        message for message in (body.get("messages") or []) if message.get("role") == "system"
    ]
    user_messages: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            user_messages.append(message)
            break

    compact: dict[str, Any] = {
        "model": body.get("model"),
        "messages": user_messages,
    }
    if system_messages:
        compact["system_blob"] = blobs.put_system(system_messages)
    tools = body.get("tools")
    if tools is not None:
        compact["tools_blob"] = blobs.put_tools(tools)
    return compact


class Recorder:
    """Appends request bodies to a JSONL file, from several threads."""

    def __init__(
        self,
        path: Path,
        compact: bool = False,
        dedupe: bool = False,
        blobs: BlobStore | None = None,
    ) -> None:
        self.path = path
        self.compact = compact
        self.dedupe = dedupe
        self.blobs = blobs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0
        self._skipped = 0
        self._seen: set[tuple[str, str]] = set()

    def write(self, record: dict) -> int:
        with self._lock:
            self._count += 1
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return self._count

    @property
    def count(self) -> int:
        return self._count

    @property
    def skipped(self) -> int:
        return self._skipped


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    recorder: Recorder
    upstream: str
    #: The key the proxy adds to the requests it forwards.  It is read from
    #: a file at start-up and never written into the job file, so the
    #: secret never lands in a file that the harness copies around.
    upstream_key: str = ""

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        pass

    def _upstream_url(self) -> str:
        path = self.path.split("?")[0]
        if path.startswith("/v1"):
            path = path[len("/v1") :]
        url = self.upstream + path
        if "?" in self.path:
            url += "?" + self.path.split("?", 1)[1]
        return url

    def _send_error(self, code: int, message: str) -> None:
        payload = json.dumps({"error": {"message": message}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream_upstream(self, body: bytes) -> None:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in SKIP_HEADERS
        }
        if self.upstream_key:
            # The proxy *replaces* the Authorization header, it does not
            # only fill in a missing one: the harness sends the key it was
            # given (``EMPTY`` for the GLM endpoint), and an upstream that
            # needs a real key would reject that.  The proxy is the one
            # place that knows the real key.
            for key in list(headers):
                if key.lower() == "authorization":
                    del headers[key]
            headers["Authorization"] = f"Bearer {self.upstream_key}"
        headers["Content-Length"] = str(len(body))
        request = urllib.request.Request(
            self._upstream_url(), data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:  # pragma: no cover -- network failure path
            self._send_error(502, str(error))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        record: dict = {"ts": time.time(), "method": "POST", "path": self.path}
        # The session id is what tells two rollouts of the *same* task
        # apart.  The dataset contains many replicas of one instruction
        # (superuser-000000_copy0000 and superuser-000000_copy0002 have
        # byte-identical instructions), so a capture that cannot attribute
        # a request to a rollout mixes their conversations -- which then
        # looks like a wire mismatch when nothing is wrong.
        session_id = self.headers.get("X-Session-ID")
        if session_id:
            record["session_id"] = session_id
        if len(body) <= MAX_RECORD_BYTES:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                record["body_text"] = body[:4096].decode("utf-8", "replace")
            else:
                if isinstance(parsed, dict):
                    recorder = self.recorder
                    if recorder.compact:
                        if recorder.dedupe and parsed.get("tools"):
                            user = _first_user_message(parsed)
                            if user is not None:
                                key = (user, _system_digest(parsed))
                                with recorder._lock:
                                    duplicate = key in recorder._seen
                                    if not duplicate:
                                        recorder._seen.add(key)
                                if duplicate:
                                    recorder._skipped += 1
                                    parsed = None
                        if parsed is not None:
                            if recorder.blobs is not None:
                                record["body"] = compact_body_with_blobs(
                                    parsed, recorder.blobs
                                )
                            else:
                                record["body"] = compact_body(parsed)
                            record["n_messages"] = len(parsed.get("messages") or [])
                    else:
                        record["body"] = parsed
                if "body" in record or "body_text" in record:
                    self.recorder.write(record)
        self._stream_upstream(body)

    def do_GET(self) -> None:  # noqa: N802
        headers = {}
        if self.upstream_key:
            headers["Authorization"] = f"Bearer {self.upstream_key}"
        try:
            request = urllib.request.Request(
                self._upstream_url(), headers=headers, method="GET"
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "application/json")
        except Exception as error:  # pragma: no cover
            self._send_error(502, str(error))
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; use 0.0.0.0 when a container has to reach it",
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get(
            "RSI_MODEL_BASE_URL_HOST", os.environ.get("MODEL_BASE_URL", "")
        )
        or "http://YOUR-ENDPOINT:PORT/v1",
        help="the endpoint the proxy forwards to, as seen from the host.  "
        "Defaults to $MODEL_BASE_URL, or $RSI_MODEL_BASE_URL_HOST when the "
        "endpoint is on the machine that runs Docker (the container reaches "
        "it at host.docker.internal, the host at 127.0.0.1).",
    )
    parser.add_argument(
        "--upstream-key-file",
        default="",
        help="a file with the API key of the upstream endpoint.  The proxy "
        "adds it as the Authorization header of the requests it forwards, so "
        "the harness never has to carry the key.  The key is read once, at "
        "start-up, and never written to the job file or to the log.",
    )
    parser.add_argument("--out", default="/tmp/opencode-requests.jsonl")
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "keep only the leading system messages, the first user message "
            "and the tool schemas of a request"
        ),
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="write one record per rollout instead of one per request",
    )
    parser.add_argument(
        "--blob-dir",
        default="",
        help=(
            "store the system prompt and the tool schemas in this "
            "directory, content-addressed, and reference them from the "
            "request log by digest.  Needs --compact.  This is what makes a "
            "production run affordable: the opencode system prompt is ~80 KB "
            "and is identical for every rollout of the same day"
        ),
    )
    args = parser.parse_args()

    blobs = None
    if args.blob_dir:
        if not args.compact:
            print("--blob-dir needs --compact", file=sys.stderr)
            return 2
        blobs = BlobStore(Path(args.blob_dir))

    recorder = Recorder(
        Path(args.out), compact=args.compact, dedupe=args.dedupe, blobs=blobs
    )
    upstream_key = ""
    if args.upstream_key_file:
        upstream_key = read_key(args.upstream_key_file)
        print(f"the upstream key comes from {args.upstream_key_file}")

    handler = type(
        "BoundHandler",
        (Handler,),
        {
            "recorder": recorder,
            "upstream": args.upstream.rstrip("/"),
            "upstream_key": upstream_key,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"recording proxy on http://{args.host}:{args.port} -> {args.upstream}")
    print(f"requests -> {args.out}", flush=True)
    print(
        f"compact={args.compact} dedupe={args.dedupe}", flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(
            f"recorded {recorder.count} requests "
            f"({recorder.skipped} duplicates skipped)"
        )
        if blobs is not None:
            print(f"blobs: {blobs.n_blobs} written to {blobs.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
