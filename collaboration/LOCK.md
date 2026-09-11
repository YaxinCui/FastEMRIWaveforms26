# Experiment edit lock

<!-- 2026-09-11 16:22 CST (linux): Start an independent master-based physical
mode/API consistency investigation. Linux root writes; Mac/agents read only. -->

- Branch: `codex/exp13-retrograde-amplitude-contract`.
- Base: master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Local/uncommitted; no host switch, no production implementation yet.
- Reuse authenticated small exp12 master archives; do not read the 5 GB table
  or run competing FEW/GPU tasks while exp12 performance tests are live.
