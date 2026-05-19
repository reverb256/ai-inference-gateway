# Contributing to ai-inference-gateway

## Permanent Principles

### Nix-native
Nix is the single source of truth. All deployment configuration lives in `nix/`. No direct k8s YAML edits, no wrapper scripts, no hand-crafted derivative configs — manifests are generated from Nix via CI/CD. If you need to change how the gateway is deployed, change the NixOS module.

### Gateway routing
All AI backend traffic goes through the gateway — circuit breakers, rate limiting, observability, and MCP brokerage depend on it. Do not route clients directly to backends (NIM, llama-cpp, vLLM, etc.). If a backend format is incompatible, fix the gateway's transformation layer, not the routing.

## Development Setup

### Prerequisites

- Nix with flakes enabled
- `nix develop` for a development shell

### Quick Start

```bash
# Enter development shell
nix develop

# Build the project
nix build

# Run checks
nix flake check

# Run tests
pytest tests/ -v
```

## Code Style

- Follow existing patterns in the codebase
- Run `nix fmt` before committing (formats via treefmt)
- All commits should pass `nix flake check`
- Python: ruff for linting, configured in pyproject.toml
- Type hints required for all new code

## Pull Request Process

1. Create a feature branch from `main` (`git checkout -b fix/description`)
2. Make your changes
3. Run checks: `nix flake check` and `pytest tests/ -v`
4. Ensure the PR template checklist is satisfied
5. Commit with clear, conventional messages
6. Push and open a Pull Request against `main`
7. Mark as ready for review (or draft if still in progress)

## Reporting Issues

- Search existing issues before creating a new one
- Use the issue templates when available
- Include: problem description, reproduction steps, expected vs actual behavior
- Label candidate issues `agent-ready` if they contain enough context for autonomous agents

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
