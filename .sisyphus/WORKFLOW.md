# Universal Orchestration Protocol (UOP)

## Overview
The Universal Orchestration Protocol (UOP) defines the structure and conventions for the `.sisyphus` directory, which serves as the Distributed Shared Memory (DSM) for the agent orchestration system.

## Directory Structure

.sisyphus/
├── state.json              # Current orchestration state (JSON)
├── plans/                  # Skill-enriched execution plans
│   ├── active/             # Currently executing plans
│   ├── completed/          # Finished plans with results
│   └── templates/          # Reusable plan templates
├── evidence/               # Verification artifacts and outcomes
├── notepads/               # Temporary working notes
└── skills/                 # Registry of available graduated skills (optional, can link to global registry)

## File Formats

### state.json
```json
{
  "version": "1.0.0",
  "last_updated": <timestamp>,
  "orchestration_state": "initialized|planning|executing|completed|failed",
  "current_plan": <plan_id or null>,
  "active_agents": [<agent_id>, ...],
  "global_context": {...} // Arbitrary key-value pairs shared across the system
}
```

### Plan Files (in plans/active/, plans/completed/, plans/templates/)
Each plan is a markdown file with the following structure:

```markdown
---
plan_id: <unique_identifier>
created_at: <timestamp>
updated_at: <timestamp>
status: draft|active|completed|failed
requirements:
  - skill: <skill_name>
    version: "<version_constraint>"
    reason: "<why this skill is needed>"
  - ...
context:
  <key-value pairs for plan execution>
steps:
  - id: <step_id>
    description: <step description>
    agent: <agent_type or specific agent identifier>
    inputs: <input specifications>
    outputs: <output specifications>
    dependencies: [<step_id>, ...] // Steps that must complete before this one
    skill_requirements:
      - skill: <skill_name>
        reason: "<why this skill is needed for this step>"
    ...
evidence:
  - <evidence_id>: <description or reference>
  - ...
notes: <free-form notes about the plan>
---
# <Plan Title>

## Overview
<Description of the plan's purpose and goals>

## Detailed Steps
<Step-by-step breakdown of the plan>

## Acceptance Criteria
<Criteria that must be met for the plan to be considered complete>

## References
<Links to related issues, documents, or resources>
```

### Evidence Files
Evidence files can be any format (markdown, JSON, images, etc.) that verify the completion of a plan or step.
They are referenced by ID in the plan's evidence section.

### Notepads
Temporary working notes for agents during execution. Notepads are not persisted beyond the session unless explicitly moved to evidence.

## Usage Guidelines

1. **State Management**: The `state.json` file should be updated by the orchestration system (e.g., kagent) at key transitions.
2. **Plan Lifecycle**: Plans start in `plans/templates/` or are created de novo. When execution begins, they are moved to `plans/active/`. Upon completion, they are moved to `plans/completed/`.
3. **Skill Registry**: The `skills/` directory can contain local skill registries, but the system primarily relies on the global skill registry via the AI Gateway.
4. **Concurrency**: Multiple agents can read from `.sisyphus/` but only the orchestrator should write to `state.json` and move plan files between directories to avoid conflicts.

## Integration with kagent and hermes

- **kagent**: During the planning phase, kagent reads the global skill registry, enriches plans with skill requirements, and writes them to `.sisyphus/plans/active/`.
- **hermes**: The hermes CLI provides commands to view the state, list plans, and examine evidence.
- **hermes-agent**: Workers read plans from `.sisyphus/plans/active/`, execute them, and update evidence and state upon completion.

## Versioning
This document describes UOP version 1.0.0.
