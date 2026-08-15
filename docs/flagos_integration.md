# FlagGems Integration Management

> R1 historical note: this document previously described the `integration/*`
> work. R2 is authoritative for current inference code and evidence:
> `docs/flagos_inference_r2_checklist.md` and
> `docs/flagos_inference_r2_report.md`.

SToFM and FlagGems are maintained as independent forks.  They are deliberately
not a Git submodule: SToFM consumes a small, versioned public API rather than a
repository checkout layout.

| Repository | Fork branch | Upstream remote | Responsibility |
| --- | --- | --- | --- |
| `priestess-bot/SToFM` | `r2/flagos-inference` | `PharMolix/SToFM` | Model bridge, extraction path, tests, and benchmark reports |
| `priestess-bot/FlagGems` | `r2/stofm-flagos-inference` | `flagos-ai/FlagGems` | Direct SToFM operator API and target adapters |

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

For coordinated local development, use the checked-out fork directly:

```bash
python -m pip install --no-build-isolation -e ../FlagGems-stofm
```

Do not update a branch name in a requirements file.  First test the candidate
FlagGems commit, push it, update the JSON lock and requirements entry, run the
SToFM tests and benchmark, then commit the SToFM lock update.

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
