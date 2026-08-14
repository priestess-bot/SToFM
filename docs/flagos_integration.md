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

The API version must equal `1`.  `flagos_backend="auto"` falls back to original
PyTorch semantics when the package is unavailable; `flagos_backend="flaggems"`
is strict and fails instead of silently using the portable fallback.

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

`--flagos_backend flaggems` enables the direct operator path.  By default,
`--flagos_attention_backend inherit` follows that selection.  Use
`--flagos_attention_backend torch` only to force the native attention path for
an A/B test.  SToFM never imports vendor extensions such as `torch_npu` or
`torch_musa`; vendor routing remains inside FlagGems.

## Upstream Sync

Fetch each upstream independently.  Integrate upstream changes into the
relevant fork branch, rerun that repository's test suite, and only then move
the SToFM lock.  An SToFM commit must always name the exact FlagGems commit it
was tested against.
