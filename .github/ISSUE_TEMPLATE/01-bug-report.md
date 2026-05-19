---
name: Bug report
description: File a bug report — something is broken or behaving incorrectly
labels: ["bug"]
---

## Describe the bug

<!-- A clear and concise description of what the bug is. -->

## Reproduction

Steps to reproduce:

1. Send request: `curl ...`
2. See error: `...`
3. ...

## Expected behavior

<!-- What should happen instead. -->

## Actual behavior

<!-- What actually happens, including full error output. -->

## Environment

- **Deployment**: [local dev / k3s dev / k3s prod]
- **Backend model**: [e.g. nvidia/nemotron-3-super-120b-a12b / llama-cpp / vLLM]
- **Client**: [e.g. curl / OpenCode / MapleSpike / Hermes]
- **Gateway version/commit**: <!-- if known -->

## Root-cause analysis (for agents)

Before implementing a fix, identify whether this is:
- **Configuration issue** → fix in `nix/options.nix` or config file
- **Gateway transformation bug** → fix in `src/` (services, middleware, router)
- **Backend incompatibility** → fix the gateway's request transformation layer, not the routing
- **Nix-native violation** → revert any raw k8s YAML changes and express through Nix

## Acceptance criteria

- [ ] Root cause identified and fixed (not worked around)
- [ ] New or updated tests cover the bug
- [ ] `nix flake check` passes
- [ ] Gateway routing preserved (no direct-to-backend bypasses)
- [ ] AGENTS.md updated if new patterns introduced

<!-- Label with `agent-ready` after triage if this contains enough context for autonomous execution. -->
