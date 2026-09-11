# exp13 — direct-retrograde amplitude API contract

<!-- 2026-09-11 16:22 CST (linux): Independent diagnostic first. Preserve
physical conventions and do not claim all retrograde source waves are wrong. -->

- Branch `codex/exp13-retrograde-amplitude-contract`, independently based on
  master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- LOCAL/UNCOMMITTED; Linux root owns editing; no host transfer.
- Production source remains master. This checkpoint prepares a reproducible
  archived-output diagnostic, not an implemented repair or speed measurement.

<!-- 2026-09-11 17:00 CST (linux): Supersede the initial diagnostic status.
Read lock, source, previous reports and exp12 producer to resume safely. -->

The initial status above is historical: the minimal Kerr-equatorial raw/all
repair is now in `src/few/amplitude/base.py`. CPU/CUDA each pass eight small
contract methods, including independently evaluated harmonic identities.
Fresh source-bound actual-table A/B validation is the remaining gate; those
small tests alone do not certify native interpolation or full waveforms.

<!-- 2026-09-11 16:29 CST (linux): Authenticated CPU/CUDA archive diagnostics
have finished. Contract failure is expected evidence, not numerical acceptance. -->

`master_cpu_contract_final.json` and `master_cuda_contract_final.json` are
terminal with `evidence_valid=true`, `bug_reproduced=true`, and
`contract_passes=false`. Each compares five modes at three p/e points. All five
tuple outputs match the stated canonical-coordinate oracle byte-for-byte;
raw-ID disagrees for four of five modes, and default all disagrees for three of
the four modes with nonnegative m. The largest observed all/tuple absolute
difference is 0.16790504 for `(2,2,0,0)`. This is not roundoff noise.
First diagnostic reports are preserved; the final ones additionally enforce
the pinned master model/frontend hashes and record producer selection/layout
source identities. No production code has been fixed yet.

## Question and concrete evidence

Does selecting the same mode by tuple, raw index, or a column of the default
all-mode output return the same physical amplitude when `xI=-1`?
The public API describes these as mode-selection alternatives, without an
exception for direct retrograde. Master `amplitude/base.py` nevertheless takes
three different sign/reindex/conjugation routes. Its tuple code even notes a
possible raw-array indexing fix.

The existing exp12 master archives contain canonical `(-a,+1)` and direct
`(+a,-1)` calls at the same p/e points, so the mismatch can be checked without
another 5 GB load or GPU job. `retrograde_archive_probe.py` authenticates the
producer and NPZ, independently enumerates mode labels, compares the five
archived selected modes across all applicable selection paths, and distinguishes
`evidence_valid` from `contract_passes`.
It preserves the producer's table/native evidence but does not freshly rehash
large data during another experiment's timed run.

## Physical hypothesis, not a generic-coordinate shortcut

For equatorial k=0, define canonical amplitudes C at `(-a,p,e,+1)` and original
coordinate amplitudes D at `(+a,p,e,-1)`. The candidate convention is

\[
D_{\ell mn}=(-1)^\ell C_{\ell,-m,n},\qquad
C_{\ell,-m,-n}=(-1)^\ell C_{\ell mn}^{*}.
\]

The first is a coordinate-axis rotation (m changes, radial n does not); the
second is the conjugate negative-mode symmetry (both m and n change).
Confusing them changes complex amplitudes, not just roundoff. Their use assumes
the corresponding phase-zero convention. It must be independently tested with
explicit spin-weighted harmonics and finite mode sums before production repair.
For m>0 this predicts `D_lmn=conj(C_l,m,-n)`, matching the present tuple route.
The finite-sum test should explicitly pair `Phi_phi'=-Phi_phi`, `Phi_r'=Phi_r`
with `(theta',phi')=(pi-theta,-phi)`, fixing polarization basis and phase zeros.
This is not a claim that the high-level API flips the supplied `Phi_phi0`.

Signed-spin table lookup in `ampinterp2d.py` is intentional and should remain.
High-level `waveform/base.py` already transforms `xI0<0` to a positive-inclination
internal convention and changes the viewing angles; it usually bypasses the
direct-amplitude discrepancy. Therefore this does NOT establish that existing
full retrograde FEW waveforms are incorrect.

## Next bounded work

1. Run the archive diagnostic on CPU and CUDA references, preserving expected
   contract failures as evidence; no speed claim.
2. Build independent analytic label tests with even/odd ell, m>0/m=0/m<0,
   n=0/positive/negative, duplicate and raw selections, scalar and array inputs.
   Explicit l=2/3 spherical-harmonic sums should check the coordinate identity.
3. On this branch only, repair raw/all direct-retrograde routing to the same
   documented convention; preserve canonical/prograde results and tuple behavior.
   Keep exp10 domain safety and exp12 residency changes on their own branches.
4. Run source-bound CPU/CUDA fixtures and actual-table comparisons. Corrected
   direct raw/all outputs should match an independently labelled canonical
   oracle, not the broken master output. Existing high-level source waveforms
   should remain unchanged. Mixed-sign xI batches/generic inclination are not
   implicitly certified by an equatorial fix.

## References

<!-- 2026-09-11 17:11 CST (linux): Fresh real-table A/B finished serially on
CPU then CUDA. Supersede the earlier pending actual-table status. -->

`retrograde_real.py` now has four terminal successful `real_*_v2.json` runs:
fresh master/fixed on CPU and CUDA. Five signed spins `0.6,-0.6,0,0.999,-0.999`,
three valid p/e points, all 6993 canonical columns, every signed raw ID (including
reversed order/repeats), seven targeted tuple modes. On BOTH backends:

- Master reproduces all/raw errors at all five spins; tuple follows the oracle.
- Fixed passes all 15 coordinate-contract checks with zero maximum difference.
- Only the expected ten direct all/raw arrays change; 24 arrays are byte-identical
  to master, including seven short high-level waveforms and their mode lists.
- Those waves cover prograde/retrograde/negative/zero/extreme spins, outer region,
  threshold selection and a positive-m explicit subset with nonzero initial phases.
  They do not cover full-waveform `all` or explicitly negative-m selection.
- All signal/amplitude arrays are complex128. Same-backend native modules and
  original H5 identities match. All imported FEW Python submodules are bound to
  master, except the single intentional `amplitude/base.py` repair.

The first CPU pilot failed AFTER numerical calls because provenance incorrectly
git-showed generated `_version.py`; preserved as `real_master_cpu.json`. The
formal v2 producer records that metadata separately. `audit_real.py` additionally
authenticates top-level `few/__init__.py`, dtypes and separate all/raw outcomes.
One producer comment says 17:06 but was authored at about 17:02 CST; runtime
JSON timestamps provide the authoritative execution order.

This completes the bounded contract repair/regression, not universal physical
truth, generic inclination, mixed-sign batches, a wheel rebuild or a speed claim.
The existing small tests also cover independent finite harmonic identities;
custom complex64 provider behavior remains without a dedicated fixture.
All changes still local/uncommitted, Linux lock retained, no host transfer.

- [Official amplitude tutorial](https://fastemriwaveforms.readthedocs.io/en/latest/tutorial/Amplitude_tutorial.html).
- [FEW Kerr paper, II.2 and III.2.1](https://arxiv.org/html/2506.09470v1):
  spherical-mode amplitudes and conjugate-mode symmetry.
- [Spherical library's spin-weighted harmonic rotation derivation](https://spherical.readthedocs.io/en/stable/SWSHs/):
  rotations must include the tangent-basis/spin-weight convention, not be
  treated as arbitrary scalar-harmonic substitutions.
- Pinned local sources: `src/few/amplitude/base.py`, `src/few/utils/baseclasses.py`,
  `src/few/waveform/base.py`, `src/few/utils/ylm.py`.
