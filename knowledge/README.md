# exp16 sources

<!-- 2026-09-11 21:11 CST (linux): Download only the matching official
fastlisaresponse native source for inspection, not building/overwriting the
installed package. Preserve upstream bytes; hashes go in experimental reports. -->

- Installed fastlisaresponse 1.1.17 records commit
  `c16cd9cbf8d69719067040cf6c170a41c696c05d`.
- [Matching native implementation](https://raw.githubusercontent.com/mikekatz04/lisa-on-gpu/c16cd9cbf8d69719067040cf6c170a41c696c05d/src/fastlisaresponse/cutils/LISAResponse.cu)
  is archived as `LISAResponse.cu` for equation/loop-order inspection.
- [Primary response paper](https://arxiv.org/html/2204.06633), section IV.3,
  equation (20): evaluating the component light-travel times at the same t is
  intentional for the GW response approximation. It is not a laser-noise
  cancellation test or fully recursive time-delay operator. Reuse must keep
  that convention, not silently replace it with different physics.
- The local newer reference repository has a different time-origin API and ABI;
  it is not the installed source oracle for this experiment.

<!-- 2026-09-11 21:34 CST (linux): Index completed derivation and byte archive.
No installed dependency, old knowledge repository or binary is modified. -->

[实验说明：可复用 TDI 算子](PREPARED_TDI.md) distinguishes cached throughput,
plan construction and response-object lifecycle. Official source archive SHA256:
`583a38e10ddee30707ba2e853c93474671d3fdfd3915ef78643d5e28e4c458c6`.
