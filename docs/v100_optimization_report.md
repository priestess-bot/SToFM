# Historical V100 Pilot Report

The original O4 pilot remains preserved in
[`benchmark-results/v100-20260815-v100-sxm2-16gb-committed`](../benchmark-results/v100-20260815-v100-sxm2-16gb-committed/).
It predates the native pair-score epilogue, vision operator work, independent
process replication, and the correction that makes the selected O5 path the
actual default CUDA inference route.

Use [`v100_operator_optimization_report.md`](v100_operator_optimization_report.md)
for the current decision, exact implementation commits, three-run raw evidence,
bootstrap confidence interval, accepted/rejected operators, and limitations.
