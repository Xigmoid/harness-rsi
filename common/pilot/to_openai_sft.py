#!/usr/bin/env python3
"""把 harness 的原生轨迹转换成 OpenAI 格式的 SFT 样本。

两种 harness 的原生记录结构完全不同，但转换后的目标结构一样：
一条 ``messages`` 列表，role ∈ {system, user, assistant, tool}，
assistant 带 ``tool_calls``，tool 带 ``tool_call_id``。

mini-swe-agent (``traj.json``)
    harness 记的 messages **逐字就是**模型侧收到的（只多两个内部键
    ``extra`` 和 ``function_call: null``）。所以这里是「剥壳」而不是重建：
      1. 去掉 ``role == "exit"`` 的消息
      2. 去掉尾部那条 ``exception_info == "action was not executed"`` 的 tool
         消息——那是提交占位符，模型从未收到过
      3. 每条剥掉 ``extra`` / ``function_call``

opencode (``session.json`` + 抓包拿到的 system prompt)
    session 是 part 树，需要重建；system prompt 不在 session 里，只能从
    记录代理抓到的请求里取。重建时 ``tool_calls[].function.arguments``
    必须用无空格 JSON（``separators=(",", ":")``）——JS 的
    ``JSON.stringify`` 不带空格，Python 的 ``json.dumps`` 默认带，
    语义等价但 token 序列不同。

用法::

    python3 to_openai_sft.py mini   <traj.json>      > out.json
    python3 to_openai_sft.py opencode <session.json> --system <system.txt> > out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

INTERNAL_KEYS = ("extra", "function_call")
NOT_EXECUTED = "action was not executed"

CJK = re.compile(r"[一-鿿]")


def repair_mojibake(text: str) -> tuple[str | None, bool]:
    """修复「UTF-8 字节被按 latin-1 解码」造成的乱码。

    实测这个损坏是**部分有损**的：多数区段是干净的双编码（可还原），但个别
    字节被替换成了空格之类的其它字符（不可还原）。所以策略是：
      1. 整串能无损往返时，直接还原
      2. 否则取「最长的可解码前缀」还原，损坏的尾巴保留原样
      3. 用第二个返回值告诉调用方尾巴是否仍有损坏

    返回 (修复后的文本, 是否仍有损坏)。不是乱码时返回 (None, False)。
    """
    if not text or not isinstance(text, str):
        return None, False
    try:
        raw = text.encode("latin-1")
    except UnicodeEncodeError:
        return None, False

    def _cjk(s: str) -> int:
        return len(CJK.findall(s))

    # 情况 1：整串可无损往返
    try:
        whole = raw.decode("utf-8")
    except UnicodeDecodeError:
        whole = None
    if whole is not None and whole != text and _cjk(whole) > _cjk(text):
        return whole, False

    # 情况 2：取最长的可解码前缀
    for end in range(len(raw), 0, -1):
        try:
            head = raw[:end].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if head == text[:end]:
            continue
        if _cjk(head) <= _cjk(text[:end]):
            continue
        tail = text[end:]
        damaged = _is_damaged(tail)
        return head + tail, damaged

    return None, False


def _is_damaged(s: str) -> bool:
    """这段文本里是否还有解不开的乱码残留。"""
    try:
        s.encode("latin-1").decode("utf-8")
        return False
    except (UnicodeEncodeError, UnicodeDecodeError):
        return True


def repair_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """就地修复所有消息里的乱码。

    返回 (消息, 警告列表, 是否有无法完全修复的样本)。
    """
    warnings: list[str] = []
    damaged = False
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        m = dict(m)
        for key in ("content", "reasoning_content"):
            val = m.get(key)
            if isinstance(val, str):
                fixed, still_bad = repair_mojibake(val)
                if fixed is not None:
                    m[key] = fixed
                    warnings.append(
                        f"fixed_mojibake@msg{i}.{key}" + ("[PARTIAL]" if still_bad else "")
                    )
                    damaged = damaged or still_bad
        out.append(m)
    return out, warnings, damaged


def _strip(msg: dict[str, Any]) -> dict[str, Any]:
    """剥掉 harness 内部键，并把 None 值字段整个去掉。

    litellm 在把消息重新序列化进下一次请求时，会省略值为 None 的字段
    （wire 上的 assistant 消息干脆没有 ``content`` 键），而 traj 里留的是
    ``content: null``。语义相同但字节不同，转换时按 wire 的形态去掉。
    """
    return {
        k: v
        for k, v in msg.items()
        if k not in INTERNAL_KEYS and v is not None
    }


def mini_to_openai(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """mini-swe-agent 的 traj.json -> OpenAI messages。"""
    out = [m for m in traj.get("messages") or [] if m.get("role") != "exit"]

    # 剥掉**末尾**那条 tool 观测：模型从来没收到过它。
    #
    # mini 的循环是「模型答 → 执行工具 → 把观测拼进 messages → 再发给模型」。
    # 所以只要 episode 在观测之后结束，那条观测就永远没有被发回去过。
    # 两种收尾都落在这一条上：
    #   * 提交指令：mini 拦下 COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 不执行，
    #     观测带 exception_info == "action was not executed"；
    #   * 超时/被切断：工具真的执行了，观测是正常的 returncode，
    #     但 episode 结束，后面没有下一次请求，模型同样没见过它。
    #
    # 剥掉之后，样本正好是「该 rollout 最后一次请求的 N 条消息」+「模型最后
    # 一次输出」= wire + 1。
    # 全部尾部观测都要剥掉，不只是最后一条：一条 assistant 消息可以带多个
    # tool call（mini 并行执行，每条 call 各追加一条观测），所以以并行
    # call 收尾的 rollout 尾部会有多条观测，而它们**都没有**被发回给模型。
    # 只剥最后一条会让前面的 call 有应答、后面的悬空——那是���样本在说模型
    # 看到了它其实没看到的东西。
    while out and out[-1].get("role") == "tool":
        out.pop()

    return [_strip(m) for m in out]


def _parts_of(msg: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [p for p in (msg.get("parts") or []) if p.get("type") == kind]


def opencode_to_openai(
    session: dict[str, Any], system_prompt: str | None = None
) -> list[dict[str, Any]]:
    """opencode 的 session export -> OpenAI messages。"""
    out: list[dict[str, Any]] = []
    if system_prompt is not None:
        out.append({"role": "system", "content": system_prompt})

    for msg in session.get("messages") or []:
        info = msg.get("info") or {}
        role = info.get("role")
        parts = msg.get("parts") or []

        if role == "user":
            text = "\n".join(p.get("text", "") for p in _parts_of(msg, "text"))
            out.append({"role": "user", "content": text})
            continue

        if role != "assistant":
            continue

        text = "\n".join(p.get("text", "") for p in _parts_of(msg, "text"))
        reasoning = "\n".join(p.get("text", "") for p in _parts_of(msg, "reasoning"))

        tool_calls = []
        for p in _parts_of(msg, "tool"):
            state = p.get("state") or {}
            tool_calls.append(
                {
                    "id": p.get("callID", ""),
                    "type": "function",
                    "function": {
                        "name": p.get("tool", ""),
                        # 无空格分隔符：对齐 wire 上 JS JSON.stringify 的形态
                        "arguments": json.dumps(
                            state.get("input") or {}, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )

        assistant: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        out.append(assistant)

        for p in _parts_of(msg, "tool"):
            state = p.get("state") or {}
            output = state.get("output")
            if output is None:
                output = (state.get("metadata") or {}).get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            out.append(
                {"role": "tool", "tool_call_id": p.get("callID", ""), "content": output}
            )

    return out


def summarize(messages: list[dict[str, Any]]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    for m in messages:
        roles[m["role"]] = roles.get(m["role"], 0) + 1
    n_calls = sum(len(m.get("tool_calls") or []) for m in messages)
    n_reason = sum(1 for m in messages if m.get("reasoning_content"))
    return {
        "n_messages": len(messages),
        "roles": roles,
        "n_tool_calls": n_calls,
        "n_assistant_with_reasoning": n_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("harness", choices=["mini", "opencode"])
    ap.add_argument("path", help="traj.json (mini) 或 session.json (opencode)")
    ap.add_argument("--system", default="", help="opencode: system prompt 文本文件")
    ap.add_argument("--task-id", default="")
    ap.add_argument("--task", default="")
    ap.add_argument("--meta", default="{}", help="额外元数据 JSON")
    args = ap.parse_args()

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))

    if args.harness == "mini":
        messages = mini_to_openai(data)
        source = "mini-swe-agent"
    else:
        system = None
        if args.system:
            system = Path(args.system).read_text(encoding="utf-8")
        messages = opencode_to_openai(data, system)
        source = "opencode"

    row = {
        "messages": messages,
        "source": source,
        "task_id": args.task_id,
        "task": args.task,
        **summarize(messages),
        **json.loads(args.meta),
    }
    json.dump(row, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
