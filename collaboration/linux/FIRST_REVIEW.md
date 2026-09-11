# First candidate review

<!-- 2026-09-11 22:06 CST (linux): Read-only peers reviewed the untested first
candidate. Root alone changes candidate and tests; no installed/native edits. -->

Before running candidate tests, reviewers found and root corrected:

- FP32/extended-precision dt/fs scalar promotion could invalidate double
  coordinate bounds or generate a longdouble array for a double native pointer.
  Convert to local Python doubles, verify their product and create explicit
  FP64 time storage. A mismatched FP32 reciprocal is rejected.
- ndarray subclasses are not the same contract: MaskedArray can mask NaNs and
  matrix.flatten stays two-dimensional. Require ordinary ndarray here.
- Odd N with fractional t0 must be judged AFTER truncating to integer cut;
  otherwise a valid one-sample interior could be refused contrary to docs.
- Bound each unit to 64 actual light-time delays and array sizes to <2**28.
  The +2 sample padding is an engineering bound in this domain, not a theorem
  for arbitrary finite FP64 values or externally rebound native geometry.
- Check output finiteness before committing state. This only catches numerical
  overflow and recoverable Python failures, not native memory faults.

This is intentionally a CPU-only experimental replacement for ONE method of
fastlisaresponse 1.1.17. It is not drop-in compatibility for every old accepted
input and is not an all-backend upstream-ready patch. CUDA's larger shared
prefetch and non-default stream ordering need an independent experiment.

Two initial fixture source comments say 21:49 CST while the preserved baseline
JSON records execution at 21:47:55 CST; those comments were manually estimated
about a minute ahead. The generated JSON timestamp is authoritative. No report
or hashed baseline source has been rewritten to hide this discrepancy.
