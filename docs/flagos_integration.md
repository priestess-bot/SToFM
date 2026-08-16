# FlagGems Integration Management

> Historical note: this document previously described only the V100 work.
> The V100 and MTT S4000 checklists are authoritative for their respective
> evidence:
> `docs/flagos_inference_r2_checklist.md` and
> `docs/musa_s4000_checklist.md`.

SToFM and FlagGems are maintained as independent forks.  They are deliberately
not a Git submodule: SToFM consumes a small, versioned public API rather than a
repository checkout layout.

| Repository | Fork branch | Upstream remote | Responsibility |
| --- | --- | --- | --- |
| `priestess-bot/SToFM` | `r2/flagos-inference` | `PharMolix/SToFM` | Model bridge, extraction path, tests, and benchmark reports |
| `priestess-bot/FlagGems` | `r2/stofm-flagos-inference` | `flagos-ai/FlagGems` | Direct SToFM operator API and target adapters |
| `priestess-bot/SToFM` | `r2/musa-s4000` | `PharMolix/SToFM` | MUSA model integration, formal experiment runner, and target evidence |
| `priestess-bot/FlagGems` | `r2/musa-s4000` | `flagos-ai/FlagGems` | MUSA native kernels, PrivateUse1 registration, and optimized backend |

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

The API version must equal `2`.  `flagos_backend="auto"` falls back to original
PyTorch semantics when the package is unavailable; `flagos_backend="flaggems"`
is strict and fails instead of silently using the portable fallback.

## Installation Modes

For a reproducible V100 environment, install
`requirements/flagos-r2-v100.txt`, followed by either the frozen stock or
optimized manifest. The exact FlagGems sources are pinned in
`deps/flagos-stock.lock.json` and `deps/flagos-optimized.lock.json`.

For MTT S4000, retain the vendor `torch` and `torch_musa` packages, then
install `requirements/flagos-musa-s4000.txt`. The exact fork commit and target
runtime contract are pinned in `deps/flagos-musa-s4000.lock.json`.

For coordinated local development, use the checked-out fork directly:

```bash
python -m pip install --no-build-isolation -e ../FlagGems-stofm
```

Do not update a branch name in a requirements file.  First test the candidate
FlagGems commit, push it, update the JSON lock and requirements entry, run the
SToFM tests and benchmark, then commit the SToFM lock update.

This commit lock is the dependency link between the two forks. A Git submodule
is intentionally unnecessary: normal installs consume the immutable Git SHA,
while adjacent editable checkouts remain available for coordinated local
development.

## Runtime Selection

`--flagos_mode` selects `torch`, `stock`, or `optimized`. Stock and optimized
modes use a temporary real `flag_gems.use_gems()` scope for the measured ATen
allowlist; only optimized mode may use R2 direct composites.
`--flagos_backend flaggems` enables the direct operator path. By default,
`--flagos_attention_backend inherit` follows that selection. SToFM never
imports vendor extensions such as `torch_npu` or `torch_musa`; vendor routing
remains inside FlagGems.

## Upstream Sync

Fetch each upstream independently.  Integrate upstream changes into the
relevant fork branch, rerun that repository's test suite, and only then move
the SToFM lock.  An SToFM commit must always name the exact FlagGems commit it
was tested against.
