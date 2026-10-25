"""Verbatim mirror of Google's reference IFEval checkers.

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
License: Apache-2.0 (Copyright 2026 The Google Research Authors)

The only modification is the import line in ``instructions.py`` (it imports the
vendored sibling module instead of the upstream package).  The mirror is used
by ``tests/test_ifeval_differential.py`` to prove that
``rsi.ifeval_checkers`` behaves exactly like the reference implementation.
"""
