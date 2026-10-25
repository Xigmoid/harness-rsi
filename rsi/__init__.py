"""Harbor adapters for the two harnesses of the SFT annotation.

The annotation drives **mini-swe-agent** and **opencode** through Harbor, and
both drivers name their agent by import path (``rsi.harbor_agent:
RsiMiniSweAgent`` / ``rsi.opencode_agent:RsiOpenCode``).  This package is
therefore the *only* piece of the host-side code that Harbor itself imports.

Three modules, and nothing else:

* :mod:`rsi.harbor_agent` -- the mini-swe-agent adapter: it pins the config
  that goes into the container and mounts the toolchain bundle.
* :mod:`rsi.opencode_agent` -- the opencode adapter: it pins the opencode
  build, skips the per-trial npm install when the image already has it, and
  exports opencode's raw session next to the trajectory.
* :mod:`rsi.paths` -- the paths the host-side scripts and the container-side
  adapters have to agree on (the toolchain bundle, and where Harbor's trial
  directories have to live so that they can be bind-mounted).

The submodules are **not** imported here on purpose: Harbor's import of the
adapter must not pull in a second copy of Harbor.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
