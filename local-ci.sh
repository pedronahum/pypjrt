#!/usr/bin/env bash
# The full run, on a box that has PJRT plugins. CI can only do tier0.
#
#   ./local-ci.sh                       # tier0 + tier1 (CPU plugin)
#   PYPJRT_GPU_PLUGIN=... ./local-ci.sh # + tier2
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
TIERS="tier0,tier1"; [[ -n "${PYPJRT_GPU_PLUGIN:-}" ]] && TIERS="$TIERS,tier2"

echo "==> pyright"
.venv/bin/pyright src/pypjrt

echo "==> abi codegen reproducible"
cp src/pypjrt/_abi/pjrt_0_114.py /tmp/pypjrt-committed.py
$PY tools/gen_abi.py >/dev/null
diff -q /tmp/pypjrt-committed.py src/pypjrt/_abi/pjrt_0_114.py \
  || { echo "FAIL: generated ABI differs from the committed one"; exit 1; }

echo "==> tests (required tiers: $TIERS)"
PYPJRT_REQUIRED_TIERS="$TIERS" $PY -m pytest -q --junitxml=/tmp/pypjrt-junit.xml

echo "==> examples"
# The examples are documentation that executes; a broken one is a broken doc.
$PY examples/run_all.py
[[ -n "${PYPJRT_GPU_PLUGIN:-}" ]] && $PY examples/run_all.py "$PYPJRT_GPU_PLUGIN"

echo "==> conformance"
CONF=()
CPU="${PYPJRT_CPU_PLUGIN:-$HOME/lib/libpjrt_c_api_cpu_plugin.so}"
[[ -f "$CPU" ]] && { $PY -m pypjrt.conform "$CPU" --json /tmp/conform-cpu.json >/dev/null; CONF+=(/tmp/conform-cpu.json); }
if [[ -n "${PYPJRT_GPU_PLUGIN:-}" ]]; then
  $PY -m pypjrt.conform "$PYPJRT_GPU_PLUGIN" --json /tmp/conform-gpu.json --memory-fraction 0.05 >/dev/null 2>&1
  CONF+=(/tmp/conform-gpu.json)
fi

echo "==> support matrix"
$PY tools/gen_support_matrix.py --junit /tmp/pypjrt-junit.xml --conform "${CONF[@]}"
echo "OK"
