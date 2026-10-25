#!/usr/bin/env python3
"""Fetch an image from Docker Hub through the *host's* proxy and load it.

``docker pull`` does not work from this machine: the Docker Desktop VM has no
route to the system proxy on 127.0.0.1:12000 (it only listens on the host's
loopback), and a direct TLS handshake to registry-1.docker.io times out often
enough that the pull fails outright.  The host *does* have a working route --
its own tools honour HTTPS_PROXY -- so the registry API is called from here
instead, and the resulting archive is handed to ``docker load``, which talks
to the daemon over a local socket and therefore needs no network at all.

Usage::

    python common/tools/fetch_image_via_registry.py \
        python:3.10-slim-trixie --platform linux/arm64

The image is loaded under its upstream name (``python:3.10-slim-trixie``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AUTH_URL = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
REGISTRY = "https://registry-1.docker.io/v2/{repo}"

#: A Docker Hub mirror that this network can reach.  ``docker pull`` cannot
#: use it (the Docker Desktop VM cannot reach registry-1.docker.io at all),
#: but the registry API over the host's proxy works, and the mirror answers
#: in ~0.1s instead of the origin's 6-25s.
MIRRORS = {
    "docker.m.daocloud.io": "https://m.daocloud.io/auth/token?service=docker.m.daocloud.io&scope=repository:{repo}:pull",
    "docker.1ms.run": "https://docker.1ms.run/openapi/v1/auth/token?service=docker.1ms.run&scope=repository:{repo}:pull",
}

MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the ``Authorization`` header when a redirect leaves the registry.

    The mirror answers a blob request with a 307 to a pre-signed object-store
    URL.  That URL carries its own signature and rejects a request that also
    carries a bearer token, and urllib would happily forward the header.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_host = urllib.parse.urlparse(req.full_url).netloc
        new_host = urllib.parse.urlparse(newurl).netloc
        if old_host != new_host:
            new.headers = {
                key: value
                for key, value in new.headers.items()
                if key.lower() != "authorization"
            }
        return new


def _opener() -> urllib.request.OpenerDirector:
    """An opener that honours the environment's proxy settings."""
    return urllib.request.build_opener(_StripAuthOnRedirect())


def _curl() -> str | None:
    """The curl binary, if there is one."""
    import shutil

    return shutil.which("curl")


def _download_blob(url: str, token: str, dest: Path) -> None:
    """Fetch one blob.

    urllib cannot follow the mirror's pre-signed redirect (it answers 403),
    so the redirect is resolved by hand and the signed URL is fetched with
    curl, which handles the query string verbatim.
    """
    curl = _curl()
    if curl is None:
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Bearer {token}")
        with _opener().open(request, timeout=900) as response:
            dest.write_bytes(response.read())
        return

    headers = subprocess.run(
        [curl, "-s", "-o", os.devnull, "-D", "-", "--max-time", "120",
         "-H", f"Authorization: Bearer {token}", url],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    location = ""
    for line in headers.splitlines():
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            break
    target = location or url
    command = [curl, "-s", "-o", str(dest), "--max-time", "1800"]
    if not location:
        command += ["-H", f"Authorization: Bearer {token}"]
    command.append(target)
    subprocess.run(command, check=True)


def _get(url: str, token: str, accept: str = "", binary: bool = False):
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    with _opener().open(request, timeout=180) as response:
        payload = response.read()
    return payload if binary else json.loads(payload)


def _token(repo: str, auth_url: str = AUTH_URL) -> str:
    with _opener().open(auth_url.format(repo=repo), timeout=120) as response:
        payload = json.loads(response.read())
    return payload.get("token") or payload["access_token"]


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def fetch(
    repo: str,
    tag: str,
    platform: str,
    outdir: Path,
    registry: str = REGISTRY,
    auth_url: str = AUTH_URL,
) -> Path:
    token = _token(repo, auth_url)
    base = registry.format(repo=repo)

    index = _get(
        base + f"/manifests/{tag}",
        token,
        accept=MANIFEST_ACCEPT,
    )
    wanted_os, wanted_arch = platform.split("/", 1)

    if "manifests" in index:
        # The index lists several arm64 entries (v8 and, on some tags, a
        # variant-less one).  Match on os/architecture and take the first hit;
        # the variant is only needed to tell arm/v7 from arm/v8 apart.
        chosen = None
        for entry in index["manifests"]:
            platform_info = entry.get("platform", {})
            if (
                platform_info.get("os") == wanted_os
                and platform_info.get("architecture") == wanted_arch
            ):
                chosen = entry
                break
        if chosen is None:
            raise SystemExit(f"no {platform} manifest in the index of {repo}:{tag}")
        digest = chosen["digest"]
        print(f"index -> {digest}", flush=True)
        manifest = _get(
            base + f"/manifests/{digest}",
            token,
            accept=MANIFEST_ACCEPT,
        )
    else:
        manifest = index

    config_digest = manifest["config"]["digest"]
    staging = outdir / "blobs"
    staging.mkdir(parents=True, exist_ok=True)

    config_target = staging / config_digest.replace(":", "_")
    if not config_target.exists():
        _download_blob(base + f"/blobs/{config_digest}", token, config_target)
    config_bytes = config_target.read_bytes()
    if _digest(config_bytes) != config_digest:
        raise SystemExit(f"config blob digest mismatch for {config_digest}")

    layers = manifest["layers"]
    print(f"{len(layers)} layers, config {config_digest[:19]}", flush=True)

    layer_paths = []
    for index_number, layer in enumerate(layers, start=1):
        blob_digest = layer["digest"]
        target = staging / blob_digest.replace(":", "_")
        if not target.exists():
            print(f"  layer {index_number}/{len(layers)} {blob_digest[:19]} ...", flush=True)
            _download_blob(base + f"/blobs/{blob_digest}", token, target)
            if _digest(target.read_bytes()) != blob_digest:
                raise SystemExit(f"layer digest mismatch for {blob_digest}")
        else:
            print(f"  layer {index_number}/{len(layers)} cached", flush=True)
        layer_paths.append(target)

    # -- the legacy docker-archive layout ---------------------------------- #
    archive_root = outdir / "archive"
    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True)

    config_name = config_digest.replace(":", "_") + ".json"
    (archive_root / config_name).write_bytes(config_bytes)

    layer_names = []
    for number, path in enumerate(layer_paths):
        name = f"{number:04d}_layer.tar"
        # The blob is a gzipped tar already; the loader detects the magic.
        shutil.copyfile(path, archive_root / name)
        layer_names.append(name)

    (archive_root / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "Config": config_name,
                    "RepoTags": [f"{repo.split('/')[-1]}:{tag}" if "/" in repo else f"{repo}:{tag}"],
                    "Layers": layer_names,
                }
            ],
            indent=2,
        )
    )

    tar_path = outdir / f"{repo.split('/')[-1]}-{tag}.tar"
    print(f"packing {tar_path} ...", flush=True)
    with tarfile.open(tar_path, "w") as tar:
        for child in sorted(archive_root.iterdir()):
            tar.add(child, arcname=child.name)
    return tar_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="repository[:tag], e.g. python:3.10-slim-trixie")
    parser.add_argument("--platform", default="linux/arm64")
    parser.add_argument(
        "--outdir",
        default="/private/tmp/rsi-harbor/image-fetch",
        help="where the blobs and the archive are kept",
    )
    parser.add_argument("--repo-tag", default="", help="override the local tag")
    parser.add_argument(
        "--mirror",
        default="docker.m.daocloud.io",
        choices=sorted(MIRRORS) + [""],
        help="a Docker Hub mirror to fetch through; '' means the origin",
    )
    args = parser.parse_args()

    repository, _, tag = args.image.partition(":")
    tag = tag or "latest"
    # "python" is short for "library/python" on Docker Hub, and the pull token
    # is scoped to the *full* name -- asking for "repository:python:pull"
    # returns a token that the registry then rejects with a 401.
    fetch_repo = repository if "/" in repository else f"library/{repository}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mirror:
        registry = f"https://{args.mirror}/v2/{{repo}}"
        auth_url = MIRRORS[args.mirror]
    else:
        registry = REGISTRY
        auth_url = AUTH_URL

    tar_path = fetch(fetch_repo, tag, args.platform, outdir, registry, auth_url)
    print(f"loading {tar_path} into docker ...", flush=True)
    result = subprocess.run(
        ["docker", "load", "-i", str(tar_path)], capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    local_tag = args.repo_tag or (
        f"{repository.split('/')[-1]}:{tag}" if "/" in repository else f"{repository}:{tag}"
    )
    subprocess.run(["docker", "tag", local_tag, args.image], check=False)
    print(f"tagged {local_tag} -> {args.image}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
