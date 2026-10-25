# The SWE-Smith family base image.
#
# This is the *family* image: every one of the 7,693 SWE-Smith tasks starts
# from this exact image, and the per-task difference (which repository, at
# which bug-injected commit) is applied at **runtime** by the task's own
# setup script.  Nothing is built per task.
#
# It exists because the published dataset's SWE-Smith rollouts ran in the
# official SWE-Smith image:
#
#   * the terminal prompt reads ``(testbed) root@...:/testbed#`` -- a conda
#     environment named ``testbed``;
#   * ``python --version`` answers 3.10.16 (6,913 of the published rollouts
#     that printed it) or 3.10.18 (2,749);
#   * the repository sits at ``/testbed`` **as the repository root**: the
#     original ``ls -la`` shows ``pypika/``, ``setup.py`` and
#     ``PyPika.egg-info`` side by side.
#
# The upstream generator agrees -- ``data/swesmith/task_templates.py`` in
# open-thoughts/OpenThoughts-Agent renders the task Dockerfile as
#
#     FROM {official_swesmith_image}
#     RUN git fetch && git checkout {instance_id}
#
# i.e. the official image plus a checkout of the instance branch.  The
# official image is amd64-only and 1.37 GB, so emulating it on this arm64
# host makes every agent command slow; this image reconstructs its
# *contents* natively and the setup script does the checkout.
#
# The conda environment is reproduced as a **venv at
# /opt/conda/envs/testbed** rather than /opt/testbed, on purpose: the shell
# prompt then reads ``(testbed)`` exactly like the original, and any command
# that reaches for ``/opt/conda/envs/testbed/bin/python`` -- the path the
# original image exposes -- keeps working.
FROM rsi/oc-base-plain:1

# git is what the upstream task Dockerfile installs on top of the official
# image; curl is what the upstream setup script uses to fetch uv.
#
# build-essential is not optional: the editable install of a repository that
# builds a C extension (pandas is built with meson) needs a compiler, and the
# official SWE-Smith image has one -- it has to, the image build runs the very
# same ``pip install -e .`` this setup script runs.  Without it the install
# dies with "Unknown compiler(s): [['cc'], ['gcc'], ...]".
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# uv 0.7.13 is the version the upstream task templates pin.
#
# ``curl ... | sh`` hides a failed download: curl's error goes to stderr, sh
# reads an empty script and exits 0, so the layer "succeeds" without uv in
# it and the next layer dies with "uv: command not found" (exit 127).  That
# is exactly what happened on the first build of this image on this network,
# so the download is retried and the pipe is checked.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN set -eux; \
    curl -fsSL --retry 8 --retry-delay 3 --retry-all-errors --retry-max-time 300 \
      https://astral.sh/uv/0.7.13/install.sh -o /tmp/uv-install.sh; \
    sh /tmp/uv-install.sh; \
    rm -f /tmp/uv-install.sh; \
    command -v uv; uv --version

# python 3.10.16 in an environment named "testbed".
#
# --seed puts pip inside the venv.  Without it ``pip`` resolves to the
# *system* pip, which refuses to install anything ("externally managed
# environment", PEP 668) -- the upstream conda environment has no such rule,
# so an agent that runs ``pip install pytest`` there succeeds.
RUN uv python install 3.10.16 \
 && uv venv --python 3.10.16 --seed /opt/conda/envs/testbed

ENV VIRTUAL_ENV=/opt/conda/envs/testbed
ENV PATH="/opt/conda/envs/testbed/bin:$PATH"

# The repository lands in /testbed at runtime; the directory has to exist
# before that, and /logs is where Harbor collects the trial's artefacts.
RUN mkdir -p /testbed /logs

WORKDIR /testbed

# uv's download cache is shared between trials through a bind mount, so the
# second task on a repository does not re-download the wheels.
ENV UV_CACHE_DIR=/var/cache/uv
RUN mkdir -p /var/cache/uv
