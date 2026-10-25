#!/usr/bin/env python3
"""Run the self-evolution study: 4 capabilities x 3 rounds + the base.

What the study produces
-----------------------
* **13 harness evaluations** (Table A).  Four capability chains -- code,
  math, reasoning, instruction following -- each evolving for three rounds
  off the same base, plus the base itself: ``4 x 3 + 1 = 13`` harnesses, each
  measured on a real benchmark with three seeds.
* **A transfer matrix** (Table B).  Every one of the 13 harnesses is also
  measured on the *other* three capabilities' benchmarks, so the matrix says
  whether the specialisation is real (a dominant diagonal) or illusory (a
  flat matrix).

The order matters.  The chains run first and the transfer probe runs
*after* them, so the transfer measurements can never leak into a chain's
optimizer history.  The optimizer is additionally restricted to the lineage
of its parent (see ``Ledger.observations(..., lineage=...)``), which makes
the two independent of each other.

    python scripts/run_study.py --chains math --rounds 3
    python scripts/run_study.py --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsi.benchmarks import STUDY_CHAINS, suite_benchmarks  # noqa: E402

#: The framework's CLI, run with the interpreter that has the package.
RSI = [sys.executable, "-m", "rsi.cli"]

#: The endpoint the agent *and* the LLM-judged verifiers talk to.
#:
#: The IneqMath verifier grades the agent's answer with an LLM
#: (``JUDGE_MODEL`` in its ``test_outputs.py``), and the OpenAI SDK reads
#: ``OPENAI_BASE_URL`` from the environment, so these two variables are
#: what make its judge reach the same endpoint as the agent.  They are set
#: here rather than left to the caller's shell, because a run that is
#: missing them does not fail loudly: the verifier catches the API error
#: and writes a reward of **-1.0**, which the collector (correctly) counts
#: as a failure -- so the run looks like a weak agent rather than a
#: misconfigured endpoint.
STUDY_ENV = {
    "OPENAI_API_KEY": "EMPTY",
    "MSWEA_API_KEY": "EMPTY",
    "OPENAI_BASE_URL": "http://YOUR-ENDPOINT:PORT/v1",
    "INEQMATH_JUDGE_MODEL": "glm5.3-flash",
}

PROPOSED_RE = re.compile(r"proposed gen (\d+)")
MEAN_RE = re.compile(r"mean=([0-9.]+)")


def _stable_hash(text: str) -> int:
    """A process-independent hash of a string.

    ``hash()`` is salted per interpreter run, so it cannot be used for a
    seed that has to survive a restart.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=_study_env())
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def _study_env() -> dict[str, str]:
    """The environment every study command runs with."""
    env = dict(os.environ)
    for key, value in STUDY_ENV.items():
        # An explicit value from the caller wins, so the study can be
        # pointed at another endpoint without editing the script.
        env.setdefault(key, value)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return env


def evaluate(gen: int, subset: str, seeds: list[int], tag: str | None = None) -> tuple[float, str]:
    cmd = RSI + [
        "gen", "evaluate",
        "--gen", str(gen),
        "--subset", subset,
        "--backend", "harbor",
        "--seeds", *[str(s) for s in seeds],
    ]
    if tag:
        cmd += ["--tag", tag]
    out = run(cmd)
    print("   " + out.strip().replace("\n", "\n   "), flush=True)
    m = MEAN_RE.search(out)
    return (float(m.group(1)) if m else float("nan")), out


def propose(parent: int, objective: str, seed: int) -> int:
    out = run(
        RSI
        + [
            "gen", "propose",
            "--parent", str(parent),
            "--objective", objective,
            "--seed", str(seed),
            "--optimizer", "bandit_ucb",
        ]
    )
    print("   " + out.strip().replace("\n", "\n   "), flush=True)
    m = PROPOSED_RE.search(out)
    if not m:
        raise SystemExit(f"could not parse the proposed generation from:\n{out}")
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", nargs="+", default=list(STUDY_CHAINS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    ap.add_argument("--transfer-seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--base-gen", type=int, default=0)
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--state", default="runs/study_state.json")
    args = ap.parse_args()

    state_path = ROOT / args.state
    state: dict = {"chains": {}, "base": {}, "home": {}, "transfer": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    def save() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    home_of = {cap: suite_benchmarks(suite)[0].key for cap, suite in STUDY_CHAINS.items()}
    suite_of = {cap: suite for cap, suite in STUDY_CHAINS.items()}

    if not args.report_only:
        # --------------------------------------------------------------- #
        # 1. the base, on every home benchmark
        # --------------------------------------------------------------- #
        for cap in args.chains:
            bench = home_of[cap]
            key = f"{cap}:{bench}"
            if key in state["base"]:
                continue
            print(f"== base on {bench} ({cap})", flush=True)
            score, _ = evaluate(args.base_gen, f"{bench}_dev", args.seeds)
            state["base"][key] = score
            save()

        # --------------------------------------------------------------- #
        # 2. the chains: three rounds of self-evolution each
        # --------------------------------------------------------------- #
        for cap in args.chains:
            suite = suite_of[cap]
            bench = home_of[cap]
            chain = state["chains"].setdefault(
                cap, {"gens": [], "suite": suite, "benchmark": bench, "scores": {}}
            )
            print(f"== chain {cap} ({suite} / {bench})", flush=True)
            parent = chain["gens"][-1] if chain["gens"] else args.base_gen
            for round_no in range(len(chain["gens"]), args.rounds):
                t0 = time.time()
                print(f" -- round {round_no + 1} (parent gen {parent})", flush=True)
                # The seed must be a *stable* function of the chain: Python's
                # `hash()` on a str is salted per process (PYTHONHASHSEED), so
                # using it would make a rerun propose a different generation
                # -- breaking the reproducibility the study is built on.
                gen = propose(parent, suite, 1000 + (_stable_hash(cap) % 997))
                score, _ = evaluate(gen, f"{bench}_dev", args.seeds)
                chain["gens"].append(gen)
                chain["scores"][str(gen)] = score
                state["home"][f"{cap}:{gen}"] = score
                save()
                print(f"    (round took {time.time() - t0:.0f}s)", flush=True)
                parent = gen

        # --------------------------------------------------------------- #
        # 3. the transfer probe
        # --------------------------------------------------------------- #
        if not args.no_transfer:
            gens = {args.base_gen: None}
            for cap in args.chains:
                for gen in state["chains"][cap]["gens"]:
                    gens[gen] = cap
            for gen, home_cap in gens.items():
                for cap in args.chains:
                    bench = home_of[cap]
                    if cap == home_cap:
                        continue
                    key = f"{gen}:{bench}"
                    if key in state["transfer"]:
                        continue
                    # The transfer probe measures the *same* (generation,
                    # benchmark) pair as the home measurement, but with a
                    # single seed.  It therefore needs its own key in the
                    # ledger and its own directory, or it would overwrite the
                    # three-seed home result and take the noise floor with
                    # it.  The tag does exactly that.
                    print(f"== transfer: gen {gen} on {bench}", flush=True)
                    score, _ = evaluate(gen, f"{bench}_dev", args.transfer_seeds, tag="transfer")
                    state["transfer"][key] = score
                    save()

    save()
    print("== study state ==")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
