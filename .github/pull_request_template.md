## Description

<!-- Describe the change and why it's needed. Link to the issue it resolves. -->

## Nix-native Compliance

- [ ] All deployment config changes expressed through Nix (nix/ directory), not raw k8s YAML
- [ ] No wrapper scripts or derivative configs hand-created (these are CI/CD outputs)
- [ ] Changes are testable via `nix flake check`
- [ ] If this modifies MCP server registration, it's done in nix/options.nix, not by hand

## Gateway Routing

- [ ] AI backend traffic routes through the gateway (not bypassed directly to providers)
- [ ] No duplicate secrets or network policy fragmentation across namespaces
- [ ] Circuit breakers and rate limiting preserved for any new backend additions

## Testing

- [ ] Tests pass: `pytest tests/ -v`
- [ ] New tests added for bug fixes or new functionality
- [ ] Manual test steps documented if applicable (for `.lan` endpoints, curl commands)

## Documentation

- [ ] AGENTS.md updated with any new agent-relevant context
- [ ] README.md or docs/ updated for significant changes
- [ ] .env.example updated if new environment variables added

## Related Issues

<!-- Closes #xxx -->
