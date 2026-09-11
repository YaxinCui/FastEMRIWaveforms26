# Project edit lock

<!-- 2026-09-01 18:24 CST (mac): Record the initial active editor and branch. -->

<!-- 2026-09-01 19:24 CST (mac): Mark the validated Mac handoff as prepared while retaining the lock until the user confirms the host switch. -->

<!-- 2026-09-01 19:27 CST (mac): Release Mac editing after the validated implementation and data commit reached GitHub. -->

<!-- 2026-09-01 19:31 CST (linux): Acquire the project edit lock after the user-directed Ubuntu pull; begin CUDA-side consistency validation. -->

<!-- 2026-09-01 20:04 CST (linux): Mark CPU/CUDA dual-host validation complete while retaining the lock until the user directs synchronization or a host switch. -->

<!-- 2026-09-01 20:10 CST (linux): Release Ubuntu editing after committing successful CPU/CUDA validation; Mac may acquire only after the user completes push and pull synchronization. -->

<!-- 2026-09-01 20:28 CST (mac): Acquire the edit lock after fast-forwarding to the completed Ubuntu handoff for Apple regression verification. -->

<!-- 2026-09-01 20:38 CST (mac): Release editing after the final Apple rebuild accepted Ubuntu's CPU/CUDA validation fixes. -->

<!-- 2026-09-01 20:38 CST (mac): Bind the released final state to the completed Mac acceptance commit before remote synchronization. -->

<!-- 2026-09-01 21:06 CST (linux): Temporarily acquire the edit lock only to document the verified high-memory Kerr table and its user-confirmed transfer to Mac. -->

<!-- 2026-09-01 21:06 CST (linux): Release the documentation-only lock so Mac can pull, verify the transferred table, and begin high-memory Kerr acceptance. -->

<!-- 2026-09-01 21:13 CST (mac): Acquire the edit lock after verifying the transferred 5.09 GB Kerr table for high-memory Apple acceptance. -->

<!-- 2026-09-01 21:29 CST (mac): Release editing after full-table Kerr amplitude/waveform acceptance and preparation of the small Ubuntu comparison artifact. -->

<!-- 2026-09-01 21:33 CST (linux): Acquire the edit lock after pulling Mac commit f7028a39 to run full-table Kerr CPU/CUDA consistency acceptance. -->

<!-- 2026-09-01 21:36 CST (linux): Mark both full-table comparisons successful while retaining the lock until the generated reports and handoff are synchronized. -->

<!-- 2026-09-01 21:38 CST (linux): Release editing after successful full-table CPU/CUDA acceptance and preparation of the user-directed report commit. -->

<!-- 2026-09-01 22:21 CST (mac): Acquire the project edit lock for the user-directed deep Apple GPU feasibility study; only isolated Metal prototypes and documented measurements may change until release. -->

<!-- 2026-09-01 22:57 CST (mac): Release the edit lock after isolating, measuring, documenting, and regression-checking the native-Metal feasibility study; changes remain local until a user-directed commit/push. -->

<!-- 2026-09-01 23:14 CST (mac): Reacquire the project edit lock for the user-directed strict-precision Metal mode-summation exploration; changes remain confined to the Mac PoC, research notes, lock, and handoff. -->

<!-- 2026-09-01 23:28 CST (mac): Release editing after the strict full-chain double-single Metal sum passed the one-year engineering gate, retained useful acceleration, and was documented and regression-checked; changes remain local until a user-directed commit/push. -->

<!-- 2026-09-01 23:30 CST (mac): Acquire the project edit lock solely for the user-directed Metal research handoff, commit, and push on the existing collaboration branch. -->

<!-- 2026-09-01 23:31 CST (mac): Release editing after the reviewed handoff set was finalized for the user-directed commit and push; the commit itself records this released state. -->

<!-- 2026-09-01 23:14 CST (linux): Acquire the edit lock to record the user-requested FP64/FP32/FP16 mixed-precision research plan and validation gates. -->

<!-- 2026-09-01 23:16 CST (linux): Mark the mixed-precision documentation complete while retaining the lock pending user-directed synchronization. -->

<!-- 2026-09-01 23:36 CST (linux): Reconcile the independently recorded lock histories after pulling Mac commit 5c872d31, preserve both hosts' work, and confirm Linux ownership for its pending documentation sync. -->

<!-- 2026-09-01 23:45 CST (linux): Release editing after documenting the mixed-precision plan and the exact missing strict-Metal artifact required from Mac. -->

<!-- 2026-09-01 23:51 CST (mac): Acquire the project edit lock after pulling Ubuntu commit 0120e06c to generate the requested strict-Metal waveform artifact/report and synchronize the existing branch. -->

<!-- 2026-09-01 23:59 CST (mac): Release editing after the five-case strict-Metal artifact/report passed integrity and regression checks and was finalized for the user-directed handoff commit/push. -->

<!-- 2026-09-02 00:04 CST (linux): Acquire the edit lock after pulling Mac commit fcfac79d to validate all five strict-Metal arrays against Linux CPU/CUDA and LPA-weighted metrics. -->

<!-- 2026-09-02 00:19 CST (linux): Mark strict-Metal CPU/CUDA and LPA-weighted validation complete while retaining the lock for user-directed review or synchronization. -->

<!-- 2026-09-02 00:36 CST (linux): Release editing after committing the passed strict-Metal validation as 1564eaee for the user-directed GitHub handoff. -->

<!-- 2026-09-02 10:53 CST (mac): Acquire the project edit lock after confirming a clean, synchronized codex/apple-silicon-dual-host worktree; isolate the remaining cross-host discrepancy by freezing the exact strict-Metal summation ABI inputs and adding a portable CPU/CUDA kernel validator. -->

<!-- 2026-09-02 11:04 CST (mac): Release editing after the frozen five-case summation inputs, Mac CPU kernel replay, integrity checks, documentation, and Ubuntu replay contract all passed; changes remain local pending a user-directed commit/push. -->

<!-- 2026-09-02 11:27 CST (mac): Reacquire the project edit lock solely for the user-directed frozen-summation handoff, commit, and push after fetching the existing branch and confirming zero local/remote divergence. -->

<!-- 2026-09-02 11:28 CST (mac): Release editing after the user-directed synchronization record and final pre-commit audit passed; this released state is included in the handoff commit. -->

<!-- 2026-09-02 11:41 CST (linux): Acquire the edit lock after fast-forwarding to Mac commit b74fb40e and verifying the frozen-input handoff identities. -->

<!-- 2026-09-02 13:32 CST (linux): Mark frozen CPU/CUDA kernel replay complete after correcting the portable validator's mixed CUDA pointer placement; retain the lock pending user-directed synchronization. -->

<!-- 2026-09-02 13:35 CST (linux): Release editing after committing the passed frozen-summation CPU/CUDA validation and CUDA ABI fix as 4d7936be for the user-directed handoff. -->

<!-- 2026-09-02 13:42 CST (mac): Acquire the project edit lock after fast-forwarding to Ubuntu handoff a7c04d81; begin the explicitly opt-in production-integration phase while preserving CPU/CUDA defaults and platform portability. -->

<!-- 2026-09-02 14:10 CST (mac): Release editing after the explicit Metal production build, five-case acceptance, wheel audit, regression suite, and Ubuntu handoff all passed; changes remain local pending a user-directed commit/push. -->

<!-- 2026-09-02 14:45 CST (mac): Reacquire the edit lock for the user-directed literature archive and whole-pipeline FEW/Apple architecture review; production backend behavior remains frozen during this research pass. -->

<!-- 2026-09-02 15:24 CST (mac): Release editing after validating the 36-PDF
local-only archive, recording its reproducible manifest and problem map, and
handing off the whole-pipeline architecture/data/error review. -->

<!-- 2026-09-02 17:31 CST (mac): Acquire the edit lock for the user-directed
P0 trajectory-reproducibility experiments, beginning with an isolated
fastmath/strict `_p_to_u` A/B and portable Mac-to-Ubuntu evidence. -->

<!-- 2026-09-02 17:56 CST (mac): Release editing after the Mac fast-math/strict
trajectory and DOP853 trace repeated bitwise, passed provenance/integrity
checks, and was documented for an authenticated Ubuntu replay. -->

<!-- 2026-09-03 15:25 CST (mac): Reacquire the edit lock solely for the
user-directed final audit, handoff record, commit, and push of the completed
Metal integration, research index, and P0 trajectory diagnostic. -->

<!-- 2026-09-03 15:29 CST (mac): Release editing after the combined handoff
payload passed remote-divergence, regression, artifact-integrity, secret, and
large-file audits; the commit containing this record is the Ubuntu sync point. -->

<!-- 2026-09-03 15:44 CST (linux): Acquire the edit lock after fast-forwarding to Mac commit c78f00dd and verifying all four production/trajectory handoff artifact identities. -->

<!-- 2026-09-03 16:24 CST (linux): Release editing after Linux AUTO/OFF portability, final-wheel CPU/CUDA regressions, frozen-sum replay, and authenticated trajectory localization all passed. -->

<!-- 2026-09-03 16:29 CST (linux): Reacquire the lock only for the user-directed hybrid CPU/Metal conclusion, handoff, commit, and push after confirming the remote still ends at c78f00dd. -->

<!-- 2026-09-03 16:30 CST (linux): Release editing after recording the accepted hybrid architecture interpretation and preparing the two-commit Ubuntu synchronization boundary. -->

<!-- 2026-09-03 17:10 CST (linux): Acquire the lock for the user-requested, source-cited 1–3 page Apple Silicon CPU adaptation report; production code and Mac-owned evidence remain read-only. -->

<!-- 2026-09-03 17:18 CST (linux): Release editing after the cited Apple Silicon CPU report passed source-link, citation, arithmetic, length, ownership, and diff audits. -->

<!-- 2026-09-03 17:31 CST (linux): Acquire the documentation lock for the user-requested 20–30 slide FEW principles-and-implementation teaching deck; production code, numerical data, and Mac-owned evidence remain read-only. -->

<!-- 2026-09-03 17:37 CST (linux): Release the documentation lock because the required presentation dependency loader is unavailable and @oai/artifact-tool cannot be resolved; no deck binary or project source was changed. -->

<!-- 2026-09-03 17:55 CST (linux): Acquire the documentation lock to record the future Mac PPT handoff and write the user-requested Apple Silicon CPU acceleration test/acceptance report; production code, numerical artifacts, and Mac-owned files remain read-only. -->

<!-- 2026-09-03 18:01 CST (linux): Release the documentation lock after recording the Mac PPT continuation and completing the cited Apple Silicon CPU test/acceptance report; citation, URL, diff, and ownership audits pass. -->

<!-- 2026-09-03 18:12 CST (linux): Acquire the documentation lock to record the user-authorized phase split: Mac owns the Apple Silicon Metal GPU explanation/PPT/test guide, while Ubuntu will later branch for an opt-in CUDA FP64/FP32 mixed-precision experiment. No branch or production source is changed in this documentation turn. -->

<!-- 2026-09-03 18:16 CST (linux): Release the documentation lock after recording the complete Mac deliverables, the explicit exception to the previous branch restriction, the proposed CUDA mixed-precision branch, and its precision/validation gates. -->

<!-- 2026-09-03 18:19 CST (linux): Reacquire the documentation lock to incorporate the user's clarification that CUDA mixed computation is outcome-driven rather than fixed to FP64+FP32; candidate precisions and CPU/GPU stage placement must be selected by measured speedup under scientific error gates. -->

<!-- 2026-09-03 18:22 CST (linux): Release the documentation lock after updating the governing mixed-compute plan and Ubuntu handoff: the winning precision/execution mix is now an experimental result selected by end-to-end speed and scientific acceptance. -->

<!-- 2026-09-03 21:32 CST (linux): Acquire the documentation lock for the user-requested Ubuntu handoff, commit, and push. The remote and local branch both resolve to e2967b67 before packaging; only the already documented reports, shared knowledge index/plan, Linux handoff, and lock are in scope. -->

<!-- 2026-09-03 21:34 CST (linux): Release editing after the six-file documentation package passed diff, timestamp-comment, ownership, report-identity, and knowledge-index audits; it is ready for the requested commit and push. -->

<!-- 2026-09-03 21:36 CST (linux): Acquire the edit lock on the newly created codex/cuda-mixed-precision branch at 4219cabf. Ubuntu begins with a read-only FP64 CUDA/type/hotspot/test audit before selecting one opt-in, reversible mixed-compute experiment. -->

<!-- 2026-09-03 22:34 CST (linux): Release editing after the first opt-in FP64 CuPy ROMAN candidate passed fresh-wheel build, focused/operator/waveform/dual-host checks, the 53-test fast CUDA suite, Ruff/format/bytecode/JSON/diff audits, and documented performance boundaries. -->

<!-- 2026-09-11 11:10 CST (linux): Acquire a documentation-only lock on the
user-selected deep-optimization branch for the requested multilayer optimization
roadmap. Earlier copied entries describe historical branches; the fields below
are the current ownership record. Research agents are read-only; Linux root is
the sole file editor. -->

<!-- 2026-09-11 11:22 CST (linux): Release the documentation-only lock after
peer review, local-link/footnote/Git-object/arithmetic checks and whitespace
validation. No source implementation, commit, push or host switch occurred. -->

- Branch: `deep-optimization`
- Active editor: `linux`
- Last editor: `linux` — Ubuntu x86_64 / NVIDIA GeForce RTX 2080 Ti
- Last lock acquired: 2026-09-11 12:57 CST (Asia/Shanghai)
- Lock released: not released
- State: Linux root owns coordination files here and isolated, master-based
  experiment worktrees in the sibling FastEMRIWaveforms-experiments directory.
  Root production source, Mac-owned files and pre-existing user scripts remain
  untouched. Research agents are read-only. No host switch is requested.

<!-- 2026-09-11 17:00 CST (linux): User requests another bounded exploration
round. Retain Linux sole-writer ownership; finish exp13 actual-table validation
and investigate phase-structured conditional likelihood in a separate master
experiment. Agents only read/research; no commit, push, or host switch. -->

<!-- 2026-09-11 12:57 CST (linux): Reacquire coordination editing for the active
user-directed independent-branch experiments. Per-worktree locks separately
identify each attempt; this entry does not switch the root checkout. -->

<!-- 2026-09-11 17:46 CST (linux): User continuation completed bounded exp15
on its own master-based worktree. Source/result reads and generated evidence
are recorded in Linux handoffs. Serial CPU jobs and 680-check audit are terminal;
root remains deep-optimization and Linux retains sole-writer ownership. No
commit/push/merge, Mac-owned edit, environment change or host switch occurred. -->

<!-- 2026-09-11 21:06 CST (linux): Revalidate active goal and authoritative
worktrees; exp15 is completed progress, no old worker is live. Start independent
master-based exp16 response preparation. Linux remains sole writer/runner;
agents read only, no installed environment/data/Mac edits or host transfer. -->

<!-- 2026-09-11 21:34 CST (linux): Exp16 source/numeric/lifecycle audit passes;
all started jobs terminal. Root coordination and new experiment notes updated
with CST, including preserved failures and narrow reuse benefits. Linux keeps
ownership; no commit/push/merge or Mac handover, overall goal remains active. -->

<!-- 2026-09-11 21:43 CST (linux): Verified exp16 progress and no live worker.
Begin independent master exp17 for dependency explicit-link input; retain Linux
sole-writer lock and leave installed packages/native/data/Mac files untouched. -->

<!-- 2026-09-11 22:17 CST (linux): Complete exp17 tests and 983-check audit;
all workers terminal, root deep-optimization unchanged. Record read/edit/output
scope in local handoff and root indexes; retain sole-writer Linux lock for open
goal. No commit/push/merge, installed source/native/H5 or Mac-owned changes. -->

<!-- 2026-09-11 22:27 CST (linux): User authorized Git publication. Seventeen
master-based experiments committed and atomically pushed, all remote hashes
verified. Coordinator documentation checkpoint follows on deep-optimization.
Linux remains sole editor; no Mac-owned edit, branch merge or host transfer. -->
