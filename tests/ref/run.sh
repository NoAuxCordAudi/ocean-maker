#!/usr/bin/env bash
# Compare the crate's Jacobian against the independent numpy reference.
# Creates a throwaway venv (numpy only) next to this script.
set -euo pipefail
cd "$(dirname "$0")/../.."
venv=tests/ref/.venv
[ -d "$venv" ] || { python3 -m venv "$venv" && "$venv/bin/pip" install -q numpy; }
out=$(mktemp --suffix=.json)
trap 'rm -f "$out"' EXIT
cargo run -q --release --example dump_reference -- "$out"
"$venv/bin/python" tests/ref/foam_ref.py "$out"
