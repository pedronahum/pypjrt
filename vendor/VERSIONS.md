# Vendored upstream sources

Pinned so the ABI we generate against is a known quantity. Re-pinning is a
codegen run (`python tools/gen_abi.py`), never an edit.

| | |
|---|---|
| Source | https://github.com/openxla/xla, branch `main` |
| Vendored | 2026-07-22 |
| PJRT C API | **0.114** (`PJRT_API_MAJOR` 0, `PJRT_API_MINOR` 114) |
| XLA FFI API | **0.3** (`XLA_FFI_API_MAJOR` 0, `XLA_FFI_API_MINOR` 3) |
| License | Apache-2.0 (see headers) |

## Contents

- `xla/pjrt/c/` — `pjrt_c_api.h`, `pjrt_c_api_macros.h`, `pjrt_c_api_device_event.h`,
  `pjrt_c_api_tpu_constants.h`, and all 19 `*_extension.h`.
- `xla/ffi/api/c_api.h` — the XLA FFI call-frame ABI.
- `xla/pjrt/proto/compile_options.proto` — field numbers for the
  `CompileOptionsProto` encoder. Parsed at codegen time; hand-typing these is how
  `use_spmd_partitioning` and `use_shardy_partitioner` are easy to get wrong.
- `CHANGELOG.md` — upstream's minor-version ledger. Track it: every delta so far
  has been additive, and a non-additive one is a re-pin event.

## Deliberately not vendored

The plugin-side implementation (`pjrt_c_api_wrapper_impl.*`, `*_internal.*`,
`pjrt_c_api_{cpu,gpu,tpu}*`) and the C++ helpers. Those are for *writing* a
plugin; we consume one. Consult them for reference semantics only.

## Version skew is negotiated, not assumed

Headers may be newer than the plugin — a caller built against newer headers
passes larger `struct_size` values and an older plugin reads only the prefix it
knows. The reverse is not safe. `pypjrt._abi.load()` hard-fails on a major
mismatch, warns on a newer minor, and selects the nearest older generated
module. Verified in practice against three ABI versions on one box: headers
0.114, CPU plugin 0.108, CUDA plugin 0.104.
