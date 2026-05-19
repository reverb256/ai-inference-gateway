---
name: Agent task
description: Well-scoped task for autonomous agent execution via Kelos
labels: ["agent-ready"]
---

## Context

<!--
Brief description of the task and why it's needed.
Link to any related issues, PRs, or discussions.

This section must contain enough context for an autonomous agent to understand
the problem without asking follow-up questions.
-->

## Scope

<!--
Define the boundaries of this task:
- What files/directories are in scope
- What is explicitly out of scope
- Any architectural constraints
-->

## Required changes

<!--
List the specific changes needed. Be concrete enough for autonomous execution:
1. Update `src/services/nim.py` to convert tool-role content format
2. Add test coverage in `tests/test_nim_format.py`
3. Register any new config in `nix/options.nix`
-->

## Clean long-term solution mandate

Agents **must** implement the cleanest long-term fix, not a workaround. Concretely:

- **Fix root causes, not symptoms** — If a backend format is incompatible, fix the gateway's transformation layer (not the routing, not the client)
- **Nix-native** — All deployment config in `nix/`. No raw k8s YAML, no wrapper scripts, no derivative configs by hand
- **Gateway-preserving** — Never route clients directly to backends. The gateway is the single entry point for all AI traffic
- **Test-first** — Include tests that prove the fix works and won't regress
- **Documentation** — Update AGENTS.md with any new patterns or requirements
- **Zero tech debt** — No `# TODO: fix later`, no workarounds, no half-measures

## Acceptance criteria

- [ ] Changes applied (list specific outcomes)
- [ ] `nix flake check` passes
- [ ] `pytest tests/ -v` passes
- [ ] Gateway routing preserved (no bypasses)
- [ ] AGENTS.md updated if new patterns introduced
- [ ] PR links back to this issue with `Closes #NNN`
