# Load the machine's configuration, and let the environment win over it.
#
# ``.env`` holds what is specific to one machine: the endpoint, the run root,
# the two interpreters, the concurrency.  It is a **default**, not an
# override -- a value that was already in the environment when the caller
# started the script wins:
#
#     RUN_ROOT=/tmp/x bash opencode_sft/run_all_stages.sh
#
# has to do what it says.  This is easy to get wrong: ``source .env`` is an
# *assignment*, and an assignment beats whatever the environment had.  The
# variables that were set beforehand are therefore saved and put back after
# the file has been read.
#
# The list below is the set of variables the pipeline reads.  Anything not in
# it is taken from ``.env`` as it is.

#: The variables a caller may override from the command line or the
#: environment.
_OVERRIDABLE=(
  MODEL_BASE_URL PROXY_UPSTREAM MODEL_NAME API_KEY API_KEY_FILE
  RUN_ROOT PY PY3 MINI_CONCURRENCY OPENCODE_CONCURRENCY MINIMAX_SHARE
)

_saved=()
for _v in "${_OVERRIDABLE[@]}"; do
  # ``eval`` on the *name* only, so the value never goes through a second
  # round of parsing; ``printf -v`` then takes the value verbatim.
  eval "_is_set=\${${_v}+1}"
  if [[ -n "${_is_set}" ]]; then
    printf -v "_val" '%s' "${!_v}"
    _saved+=("$_v=$_val")
  fi
done

if [[ -f "${REPO:-.}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO:-.}/.env"
  set +a
fi

for _kv in ${_saved[@]+"${_saved[@]}"}; do
  export "${_kv?}"
done

unset _saved _v _is_set _val _kv _OVERRIDABLE
