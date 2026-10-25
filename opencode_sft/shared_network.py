"""The one Docker network that every trial of the re-annotation shares.

Why this exists
---------------
Harbor gives every trial its own compose project, and every compose project
its own bridge network.  Those networks come out of Docker's **default
address pool**, which is ``172.17.0.0/12`` carved into /24s -- sixteen
networks by default.  A trial that ends cleanly takes its network with it, but
a trial that fails, or a job that is cancelled, leaks it.  The leaked
networks are not reusable, and after a few hundred rollouts the pool is
exhausted and **every new trial dies** with::

    failed to create network ...: all predefined address pools have been
    fully subnetted

which looks exactly like an agent failure -- the trial raises a
``RuntimeError`` from the environment, not from the model -- and is not one.

Widening the pool the obvious way does not work here: putting
``default-address-pools`` into ``~/.docker/daemon.json`` makes Docker
Desktop 4.38.0 on this host fail to start its engine altogether (the VM comes
up, dockerd never answers the API).  So the pool is left alone and the leak is
removed at the other end: **every trial joins one network that is created
once and never taken down**.

The network is declared ``external``, so ``docker compose down`` -- which is
what Harbor runs when it cleans a trial up -- leaves it alone.

Is one network safe for concurrent trials?
------------------------------------------
Yes, because nothing resolves a name on it.  Harbor reaches the container
with ``docker compose exec``, which finds the container by its compose
labels, not by DNS; the containers get the network alias ``main``, which
would be ambiguous, but nothing ever looks it up.  The agent talks to the
model through ``host.docker.internal``, which does not go through the
network at all.
"""

from __future__ import annotations

import subprocess

#: The name of the network every trial joins.
SHARED_NETWORK = "rsi-shared"

#: The compose fragment that points a trial at the shared network.
#:
#: It is merged into the trial's own ``environment/docker-compose.yaml``, and
#: it declares the network **external**, so Compose uses the existing one
#: instead of creating a per-project network.
NETWORK_COMPOSE_YAML = """# Every trial of the re-annotation shares one Docker network.
#
# Harbor gives each trial its own compose project and therefore its own
# bridge network, and those come out of a pool of sixteen.  A trial that
# fails leaks its network, and once the pool is empty every later trial
# fails to start -- which looks like an agent failure and is not one.  The
# network is declared external so that "docker compose down", which Harbor
# runs when it cleans a trial up, does not take it away.
networks:
  default:
    name: {name}
    external: true
"""


def compose_snippet(name: str = SHARED_NETWORK) -> str:
    """The compose fragment, as YAML."""
    return NETWORK_COMPOSE_YAML.format(name=name)


def ensure_shared_network(name: str = SHARED_NETWORK) -> bool:
    """Create the shared network if it is not there.

    Returns True when the network exists afterwards.  The driver calls this
    before every batch, so a network that was removed by hand (or by a
    ``docker network prune``) is recreated instead of failing the batch.
    """
    check = subprocess.run(
        ["docker", "network", "inspect", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        return True
    create = subprocess.run(
        ["docker", "network", "create", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return create.returncode == 0


def remove_leaked_networks(keep: str = SHARED_NETWORK) -> int:
    """Remove the per-trial networks that leaked; keep the shared one.

    ``docker network prune`` cannot be used: it would take the shared network
    with it whenever no container happens to be attached to it, which is most
    of the time between batches.  So the leaked networks are removed by name.
    """
    listing = subprocess.run(
        ["docker", "network", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    removed = 0
    for network in listing.stdout.splitlines():
        network = network.strip()
        if not network or network == keep:
            continue
        # Harbor's per-trial networks are the compose project's default
        # network; the shared one is the only network without that suffix.
        if not network.endswith("_env_default"):
            continue
        outcome = subprocess.run(
            ["docker", "network", "rm", network],
            capture_output=True,
            text=True,
            check=False,
        )
        if outcome.returncode == 0:
            removed += 1
    return removed
