---
name: Feature request
description: Suggest a new feature or enhancement
labels: ["enhancement"]
---

## Problem statement

<!-- What problem does this feature solve? Who benefits and how? -->

## Proposed solution

<!-- Describe the feature — what it does, how it works, high-level design. -->

## Alternatives considered

<!-- What other approaches were considered and why was this one chosen? -->

## Clean long-term solution checklist

Agents implementing this must ensure:

- [ ] **Nix-native**: All deployment config in `nix/`, not raw k8s YAML
- [ ] **Gateway routing**: New backends or features route through the gateway, not around it
- [ ] **Root cause, not workaround**: If this exists to work around a limitation, fix the limitation instead
- [ ] **Test coverage**: Unit tests for new logic, integration tests for backend interaction
- [ ] **Documentation**: AGENTS.md updated with new patterns, endpoints, or backends
- [ ] **No tech debt**: No TODOs left in code, no `# TODO: fix this later` comments
- [ ] **Backwards compatible**: Existing clients and configs continue working without changes

## Related issues

<!-- Link to related issues, PRs, or discussions. -->
