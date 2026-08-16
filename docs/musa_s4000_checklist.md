# MTT S4000 MUSA Execution Checklist

This checklist is the single source of truth for the native MUSA validation of
the SToFM Gaussian pair-bias and pair-score epilogue operators. Update an item
immediately after its command, test, or artifact has been completed. Do not
record a performance claim until its raw samples and correctness evidence exist.

## Source Control

- [x] Create `r2/musa-s4000` from the current V100-validated branch in both forks.
  Evidence: FlagGems starts at `399d0381ed63a79018f3112ecc43894fd58ba052`;
  SToFM starts at `7eecbe9809ecfceebc50c772cc8720677995dca6`.
- [x] Commit and push the FlagGems MUSA build, dispatch, kernel, and target
  tests before advancing the SToFM dependency.
  Evidence: `priestess-bot/FlagGems:r2/musa-s4000` at immutable commit
  `832c46df4073215d416406181484f9b44594aff2`.
- [x] Lock the pushed FlagGems SHA, then commit the SToFM integration, tests,
  formal benchmark runner, and reports.
  Evidence: `deps/flagos-musa-s4000.lock.json` pins FlagGems `832c46df`; the
  tested SToFM implementation is `e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4`.
- [x] Push the SToFM `r2/musa-s4000` implementation branch.
  Evidence: `priestess-bot/SToFM:r2/musa-s4000` contains `e2c6de9e`.
- [x] Record both immutable implementation SHAs in every formal result.
  Evidence: all 15 formal result JSON files record SToFM `e2c6de9e`, FlagGems
  `832c46df`, and the expected initial or optimized native-library SHA-256;
  the independent artifact verifier passed.
- [ ] Publish the final evidence commit.
  publish the final evidence commit.

## Target Preconditions

- [x] Confirm the target host has one MTT S4000 with 47.91 GiB memory and
  `torch.musa.is_available() == True`.
- [x] Capture the target software contract: MUSA 3.1.0, `torch` 2.2.0,
  `torch_musa` 1.3.0, Python 3.10, and device architecture `mp_22`.
- [x] Create an isolated target virtual environment that inherits the vendor
  MUSA packages and installs only SToFM test dependencies.
  Evidence: `/root/stofm-musa-r2/.venv` was created with
  `--system-site-packages`; `transformers==4.39.1` and `pytest` installed on
  2026-08-16 without replacing the target `torch` or `torch_musa` packages.
- [x] Fetch the exact two fork SHAs to the target host and verify source hashes.
  Evidence: fresh, clean target checkouts resolved to FlagGems `832c46df` and
  SToFM `e2c6de9e`; the target suites then passed 25/25 and 2/2 respectively.

## Native MUSA Extension

- [x] Add the supported `torch_musa` build entry point and remove the invalid
  `mp_31` release-build assumption.
  Evidence: `vendor/musa_stofm/setup.py`; offline deferred-native validation
  passed on 2026-08-16.
- [x] Add lazy extension loading, explicit native-required failure mode, and
  `PrivateUse1` dispatch verification.
  Evidence: `stofm_backends/mthreads.py`; target execution remains pending.
- [x] Build the extension on the S4000, load it through `torch.ops.load_library`,
  and retain the build log, library hash, and dynamic-link evidence.
  Evidence: forced build on 2026-08-16 produced
  `flagos_stofm_musa.so` with SHA-256
  `006d5e256060342f1fb188f91e623fed0baaa0746928d710a43de63efb1cf590`;
  `torch.ops.load_library` registered `PrivateUse1` kernels and the target
  artifact directory retains `build.log` and `build_manifest.json`.

## Operator Correctness

- [x] Implement and validate the FP32-accumulating Gaussian pair-bias kernel.
  Evidence: S4000 native test passed for FP32, FP16, and BF16 on 2026-08-16.
- [x] Implement and validate the parallel pair-score softmax/context kernel.
  Evidence: S4000 native test passed all mask and optional-output combinations
  on 2026-08-16; broader precision and contract coverage remains below.
- [x] Replace the first serial MUSA kernels with cooperative implementations
  and preserve the original binary as a separate measurement baseline.
  Evidence: the cooperative Gaussian kernel assigns one 128-thread hardware
  warp to each pair; pair-score uses hierarchical reduction and parallel
  context accumulation. The forced S4000 build produced SHA-256
  `ba3d96f06c6107c103385c3df635b9129475db7c647b3c46ded8a02358f8c974`;
  the serial binary remains retained under the target artifact tree.
- [x] Remove nested PrivateUse1-to-ATen scheduling gaps from the FP32 Gaussian
  implementation while preserving the public operator API.
  Evidence: the final MUSA extension registers internal RBF and layout
  primitives and lets the FlagOS backend orchestrate the vendor linear
  kernels. Its S4000 binary SHA-256 is
  `a7beac88e8d4b7b999b3620b13234ded848aee22d1a479f66ce9f9744a8e2313`.
  On the N=1050 preflight, the public backend reduced Gaussian p50 from
  22.941 ms to 9.594 ms (2.39x; 58.2% lower latency) with maximum absolute
  error 1.49e-7. The same binary retained the pair-score SDPA fast path.
- [x] Pass direct native-op tests for FP32, FP16, and supported BF16 cases.
  Evidence: target `tests/test_musa_stofm_native.py` completed 25/25 on
  2026-08-16. This includes the public FlagOS backend in FP32, FP16, and BF16.
  BF16 pair-score checks use a CPU FP32 oracle fed the same BF16 input values.
  The optimized BF16 SDPA route reached a 0.00351 maximum absolute error in
  the N=1050 preflight, below the 5e-3 absolute test tolerance.
- [x] Pass masks, optional outputs, non-contiguous fallback, invalid-contract,
  and inference-only/autograd fallback tests.
  Evidence: the same 25/25 run exercised all four `(return_pair,
  return_weights)` combinations, padding, non-contiguous reference fallback,
  native-required behavior, autograd rejection, and invalid mask dtype checks.
- [x] Pass end-to-end SToFM equality and dispatch-provenance tests.
  Evidence: target `tests/test_musa_s4000_target.py` passed 2/2 on
  2026-08-16. It compares full `last_hidden_state` and `pair_rep` against the
  same-weight PyTorch model for both unpadded and genuinely padded inputs, and
  proves both Gaussian and every tested pair-attention boundary selected
  `mthreads` with registered PrivateUse1 kernels.

## Performance Evidence

- [x] Run the 64-node smoke workload before the full experiment.
  Evidence: target worker completed the four correctness-gated FP32 stages at
  N=64, two layers, four heads on 2026-08-16 and wrote raw MUSA-event and host
  samples. All four validation gates passed; small-shape latency was retained
  as a smoke result only, not promoted as a performance claim.
- [x] Correct the benchmark inference-mode protocol and invalidate timing rows
  collected through the Autograd reference dispatch.
  Evidence: both MUSA benchmark workers now wrap the complete warmup and timed
  regions in `torch.inference_mode()`. Corrected N=1050 direct measurements
  exposed the true initial serial-kernel costs (2667.10 ms Gaussian and
  9.616 ms pair-score); earlier approximately 24 ms rows were reference
  fallbacks and are explicitly excluded from final claims.
- [x] Demonstrate corrected end-to-end gain before starting formal trials.
  Evidence: the N=1050, four-layer FP32 preflight measured 33.099 ms for pure
  PyTorch, 19.884 ms with only Gaussian optimization, 29.540 ms with only
  pair-attention optimization, and 16.267 ms with both operators. The combined
  path is 2.03x faster (50.85% lower p50 latency), with maximum absolute model
  error 1.19e-6.
- [x] Validate the optimized public backend for FP16 and BF16 before the full
  precision matrix.
  Evidence: the N=1050 preflight measured Gaussian speedups of 2.36x in FP16
  and 2.37x in BF16. Pair-attention measured 1.75x in FP16 and 1.88x in BF16;
  its BF16 maximum absolute error against the CPU FP32 oracle was 0.00351.
- [x] Run the 1050-node, four-layer primary workload for pure PyTorch, frozen
  stock FlagOS, each native operator, combined native operators, and combined
  native plus FlagOS ATen dispatch.
  Evidence: five independent trials measured pure PyTorch at 33.2088 ms,
  Gaussian-only at 19.9878 ms, pair-attention-only at 29.5774 ms, and both
  operators at 16.3370 ms. Generic FlagOS ATen dispatch and frozen upstream
  FlagOS were retained as unavailable rows because MUSA Triton has no driver.
- [x] Run the `N=256/512/1050/2048` shape matrix for supported precisions.
  Evidence: five trials covered both operators at all 12 FP32/FP16/BF16 shape
  combinations, producing 120 correctness-gated optimized measurement rows.
- [x] Aggregate five independent runs per stage with raw samples, percentiles,
  variation, and bootstrap confidence intervals.
  Evidence: `artifacts/musa_s4000/formal-20260816` contains all raw device and
  host samples. The combined model speedup is 2.032x with a paired hierarchical
  bootstrap 95% interval of [2.028x, 2.036x] and 0.28% trial-p50 variation.
- [x] Record stock-baseline ABI availability without substituting a different
  FlagOS version if the frozen baseline cannot run on MUSA 3.1.
  Evidence: frozen FlagGems `03bf364ede763d573d5c30124d554283a209ab85`
  was probed on the S4000 with the same `torch`/`torch_musa` runtime. It is
  unavailable because generic Triton reports `0 active drivers`; the raw probe
  result and traceback were retained rather than substituting another version.

## Reporting

- [x] Generate Markdown and human-readable HTML reports with full baseline
  names, per-operator charts, correctness matrix, and raw-artifact links.
  Evidence: `docs/musa_s4000_optimization_report.md` and the integrated
  `docs/flagos_inference_r2_report.html` include three-way operator results,
  end-to-end charts, the 24-case precision/shape matrix, optimization history,
  code links, test conclusions, limitations, and raw evidence links.
- [x] Review the HTML at desktop and mobile widths and verify that it exposes no
  local path, target address, or credential.
  Evidence: Playwright 1.62.1 passed at 1440x1000 and 390x844 with zero page
  overflow, console warnings, console errors, or page exceptions. FP16/BF16 tab
  interaction updated all eight matrix rows; source scanning found no local
  code path, target endpoint, or credential. Browser plugin was not available,
  so the documented Playwright fallback was used.
