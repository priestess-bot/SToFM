# FlagGems Integration Management

SToFM and FlagGems are maintained as independent forks.  They are deliberately
not a Git submodule: SToFM consumes a small, versioned public API rather than a
repository checkout layout.

| Repository | Fork branch | Upstream remote | Responsibility |
| --- | --- | --- | --- |
| `priestess-bot/SToFM` | `integration/flagos` | `PharMolix/SToFM` | Model bridge, extraction path, tests, and benchmark reports |
| `priestess-bot/FlagGems` | `integration/stofm` | `flagos-ai/FlagGems` | Direct SToFM operator API and target adapters |

## Dependency Contract

The SToFM bridge imports only these FlagGems public symbols:

```python
from flag_gems.experimental_ops import (
    STOFM_EXPERIMENTAL_API_VERSION,
    resolve_stofm_backend,
    stofm_gaussian_pair_bias,
    stofm_pair_attention,
)
```

The API version must equal `1`. `flagos_backend="auto"` falls back to original
PyTorch semantics when the package is unavailable; `flagos_backend="flaggems"`
is strict and fails instead of silently using the portable fallback. Explicit
`inductor`, `nvidia`, `ascend`, and `mthreads` selections are also accepted by
the bridge. The latter two require an `npu`/`musa` tensor respectively and are
kept vendor-import-free until their rented-device runtime is installed.

## Installation Modes

For a reproducible V100 environment, install
`requirements/flagos-v100.txt`.  The FlagGems source is pinned by full commit
in both that file and `deps/flagos.lock.json`.

For coordinated local development, use the checked-out fork directly:

```bash
python -m pip install --no-build-isolation -e ../FlagGems-stofm
```

Do not update a branch name in a requirements file.  First test the candidate
FlagGems commit, push it, update the JSON lock and requirements entry, run the
SToFM tests and benchmark, then commit the SToFM lock update.

## Runtime Selection

`--flagos_backend flaggems` enables the selected CUDA inference path. With the
default `--flagos_attention_backend inherit`, Gaussian pair bias resolves to
the CUDA Inductor O1 path and pair-state attention resolves to the NVIDIA O2n
epilogue when its inference preconditions hold. Gradient-enabled, dropout,
unsupported-dtype, and unsupported-layout calls retain the public reference
semantics.

Use `--flagos_attention_backend torch` for the O3 comparison path and
`--flagos_attention_backend inductor` for the O4 direct pair-reference path.
`--flagos_backend nvidia` is an explicit native-kernel experiment, not the
recommended default, because its Gaussian O1n kernel is slower than O1 on
V100. `ascend` and `mthreads` select the correctness-first target adapters on
their matching device types. SToFM never imports vendor extensions such as
`torch_npu` or `torch_musa`; vendor routing remains inside FlagGems.

## Upstream Sync

Fetch each upstream independently. Integrate upstream changes into the
relevant fork branch, rerun that repository's test suite, push the tested
FlagGems SHA, then move `deps/flagos.lock.json` and
`requirements/flagos-v100.txt` in SToFM. Run the SToFM tests and benchmark
against that immutable SHA before pushing the SToFM integration commit. An
SToFM report must name the exact FlagGems commit it was tested against; a
branch name alone is never a dependency version.
