#!/usr/bin/env python3
"""End-to-end smoke test of the RHI harness against a live model.

Runs the real ``RhiAgent`` (multi-hop, contract-filtered context) in a local
environment and prints the transcript of the run, so the delegation protocol
and the contract extraction can be inspected by eye.

    python scripts/smoke_rhi.py --task "create /tmp/rhi_smoke/hello.txt with 'hi'"

It uses the project's frozen base config (harness/base_config.yaml) and the
gen-0 spec, so it exercises exactly the code path the benchmark uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from minisweagent.environments import get_environment  # noqa: E402
from minisweagent.models import get_model  # noqa: E402

from rsi.adapter import to_mini_config  # noqa: E402
from rsi.ledger.ledger import Generation  # noqa: E402
from rsi.ledger.diff import diff_specs  # noqa: E402
from rsi.rhi_agent import RhiAgent  # noqa: E402
from rsi.schema import load_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--gen", type=int, default=0)
    parser.add_argument(
        "--task",
        default="Create the file /tmp/rhi_smoke/hello.txt containing exactly the text `hi`, "
        "then finish.",
    )
    parser.add_argument("--spec", default=None, help="path to a spec file (default: harness/genNN.yaml)")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--out", default=None, help="where to write the trajectory")
    args = parser.parse_args()

    root = Path(args.root)
    spec_path = Path(args.spec) if args.spec else root / "harness" / f"gen{args.gen:02d}.yaml"
    spec = load_spec(spec_path)
    base = yaml.safe_load((root / "harness" / "base_config.yaml").read_text(encoding="utf-8"))

    config = to_mini_config(spec, base, spec_path=spec_path)
    # The local environment keeps the smoke test runnable without Docker.
    config["environment"] = {
        "environment_class": "local",
        "cwd": "/tmp",
        "timeout": 60,
    }
    config["agent"]["cost_limit"] = 0.0
    config["agent"]["wall_time_limit_seconds"] = 900
    if args.out:
        config["agent"]["output_path"] = str(Path(args.out).resolve())

    model = get_model(config=config["model"])
    env = get_environment(config["environment"], default_type="local")
    agent = RhiAgent(model, env, **config["agent"])

    print(f"spec        : {spec_path}")
    print(f"sha256      : {spec.spec_sha256()}")
    print(f"orchestrator: {spec.orchestrator}")
    print(f"hops        : {[(h.id, h.source, h.target, h.pass_fields) for h in spec.workflow]}")
    print(f"task        : {args.task}")
    print("-" * 78)

    result = agent.run(args.task)

    print("-" * 78)
    print(f"exit_status : {result.get('exit_status')}")
    print(f"submission  : {str(result.get('submission'))[:300]}")
    if agent.state is not None:
        summary = agent.hop_runner.summary(agent.state)
        print(f"cycles      : {summary['cycles']}")
        print(f"turns       : {summary['turns']}")
        print(f"llm calls   : {summary['n_calls']}")
        print(f"parse fails : {summary['parse_failures']}")
        print("hop log     :")
        for hop in summary["hops"]:
            print(
                f"  {hop['hop_id']}: {hop['source']} -> {hop['target']} "
                f"fields={hop['fields']} turns={hop['n_turns']} "
                f"degraded={hop['degraded']} error={hop['error']}"
            )
    print("-" * 78)
    print("transcript:")
    for msg in agent.messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        agent_id = (msg.get("extra") or {}).get("rhi_agent_id", "-")
        print(f"[{role:<9}][{agent_id:<13}] {content[:400]}")

    if args.out:
        print(f"\ntrajectory written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
