"""RHI: a typed-configuration framework for recursive harness self-improvement.

The package implements the upgrade of the RHI recipe described in
``01-research-report.md`` / ``02-coding-plan.md``: instead of letting an LLM
rewrite the harness prompt, a typed configuration is searched and every
generation is measured on a benchmark with paired statistics.
"""

from .schema import (
    AgentSpec,
    AuxiliaryRules,
    Contract,
    ContractFormat,
    FailurePolicy,
    GoalStyle,
    HarnessSpec,
    Hop,
    InstructionSlots,
    Runtime,
    SpecError,
    VerificationStyle,
    load_spec,
    save_spec,
)

__version__ = "0.1.0"

__all__ = [
    "AgentSpec",
    "AuxiliaryRules",
    "Contract",
    "ContractFormat",
    "FailurePolicy",
    "GoalStyle",
    "HarnessSpec",
    "Hop",
    "InstructionSlots",
    "Runtime",
    "SpecError",
    "VerificationStyle",
    "load_spec",
    "save_spec",
    "__version__",
]
