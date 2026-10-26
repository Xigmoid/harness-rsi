#!/usr/bin/env python3
"""把 5 个任务的原生轨迹转成 OpenAI 格式 SFT 样本，并用抓包逐条校验。

两个 harness 完全隔离，各自一个抓包代理、一份抓包文件：
    captures/mini-wire.jsonl      <- 8897
    captures/opencode-wire.jsonl   <- 8898

校验规则（两个 harness 同构）：
    转换结果的前 N 条 == 该 rollout 最后一次请求的 N 条消息
    （N = 最后一次请求的消息数），且转换结果比它多出恰好 1 条
    —— 那一条就是模型最后一次输出（从未被发回去过）。

opencode 侧还额外容忍一处差异：``tool_calls[].function.arguments``
的 JSON 空格（wire 用 JS JSON.stringify，无空格）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_openai_sft import (  # noqa: E402
    mini_to_openai,
    opencode_to_openai,
    repair_messages,
    summarize,
)

#: Where this script lives.
HERE = Path(__file__).resolve().parent
#: Where the pilot's artefacts are: the captures and the raw harness output.
#: They are generated, so they are not in the repository.
ROOT = HERE.parent.parent / "work" / "pilot" / "terminal-5task"
TITLE_MARK = "Generate a title for this conversation"


def norm_toolcall_args(msgs):
    """把 tool_calls 的 arguments 统一成无空格 JSON，便于与 wire 比对。"""
    out = []
    for m in msgs:
        m = dict(m)
        tcs = m.get("tool_calls")
        if tcs:
            new = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.dumps(
                            json.loads(fn["arguments"]), ensure_ascii=False, separators=(",", ":")
                        )
                    except json.JSONDecodeError:
                        pass
                tc["function"] = fn
                new.append(tc)
            m["tool_calls"] = new
        out.append(m)
    return out


def key_of(msg_list):
    for m in msg_list:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def load_capture(name):
    path = ROOT / "captures" / f"{name}-wire.jsonl"
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def compare(converted, wire, strict):
    """转换结果的前 len(wire) 条应与 wire 逐条相同；转换结果应多出恰好 1 条。"""
    n = len(wire)
    if len(converted) != n + 1:
        return False, f"长度应为 wire+1={n+1}，实际 {len(converted)}"
    a = converted[:n]
    b = wire if strict else norm_toolcall_args(wire)
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                dx = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
                return False, f"[{i}] role={x.get('role')} 差异字段 {sorted(dx)}"
        return False, "前缀不等"
    return True, f"前 {n} 条与 wire 逐字一致，多出 1 条 role={converted[-1].get('role')}"


def main() -> int:
    tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    caps = {"mini": load_capture("mini"), "opencode": load_capture("opencode")}

    report = []
    for t in tasks:
        tid, slug, task_text = t["id"], t["slug"], t["task"]
        per = {}
        for name, recs in caps.items():
            hits = []
            for i, r in enumerate(recs):
                k = key_of(r["body"]["messages"])
                if TITLE_MARK in k or task_text[:18] not in k:
                    continue
                hits.append((i, r["body"]))
            per[name] = hits
            if not hits:
                print(f"!! {tid} {slug}: {name} 抓包缺失")

        if not per["mini"] or not per["opencode"]:
            continue

        # ---- mini ----
        mini_wire = per["mini"][-1][1]["messages"]
        mini_msgs = mini_to_openai(json.loads((ROOT / "mini" / "raw" / f"{tid}.json").read_text(encoding="utf-8")))
        # 先与 wire 比对（wire 本身可能带乱码），比对通过后再做乱码修复
        mini_ok, mini_note = compare(mini_msgs, mini_wire, strict=True)
        mini_msgs, mini_warn, mini_bad = repair_messages(mini_msgs)

        # ---- opencode ----
        oc_wire = per["opencode"][-1][1]["messages"]
        system_prompt = oc_wire[0].get("content") if oc_wire and oc_wire[0]["role"] == "system" else None
        oc_msgs = opencode_to_openai(
            json.loads((ROOT / "opencode" / "raw" / f"{tid}.json").read_text(encoding="utf-8")), system_prompt
        )
        oc_ok, oc_note = compare(oc_msgs, oc_wire, strict=False)
        oc_msgs, oc_warn, oc_bad = repair_messages(oc_msgs)

        for name, msgs, ok, note, warn, bad, wire in (
            ("mini", mini_msgs, mini_ok, mini_note, mini_warn, mini_bad, mini_wire),
            ("opencode", oc_msgs, oc_ok, oc_note, oc_warn, oc_bad, oc_wire),
        ):
            row = {
                "messages": msgs,
                "source": name,
                "task_id": tid,
                "task_slug": slug,
                "task": task_text,
                "n_wire_requests": len(per[name]),
                **summarize(msgs),
                "wire_parity": {"ok": ok, "note": note},
                "warnings": warn,
                "usable_for_sft": not bad,
            }
            out = ROOT / name / "sft" / f"{tid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            report.append((tid, name, ok, note, warn, bad, len(msgs), len(wire)))

    print(f"\n{'id':<4}{'harness':<10}{'wire校验':<10}{'条数':>5}{'wire':>6}  备注")
    for tid, name, ok, note, warn, bad, n, w in report:
        wtxt = f"  ⚠ {','.join(warn)}" + (" [仍有损坏→不建议入训练集]" if bad else "") if warn else ""
        print(f"{tid:<4}{name:<10}{'✓' if ok else '✗':<10}{n:>5}{w:>6}  {note}{wtxt}")
    bad = [r for r in report if not r[2]]
    print(f"\n共 {len(report)} 条，全部通过: {not bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
