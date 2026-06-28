---
name: commit
description: Create commits using Conventional Commits format. Use when the user asks to commit changes.
---

# Commit Skill

Creates commits following [Conventional Commits](https://www.conventionalcommits.org/).

## Format

```
<type>[optional scope][!]: <description>

[optional body]

[optional footer(s)]
```

**Rules:**
- First line short, ideally 72 characters or less
- Use imperative mood ("Add feature" not "Added feature")
- No scope by default; only when disambiguation needed (e.g., `fix(parser):`)
- Breaking changes: add `!` before `:` AND include `BREAKING CHANGE:` in footer

## Types

- `feat`: New feature
- `fix`: Bug fix
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `chore`: Maintenance (deps, config, build)
- `docs`: Documentation

## Process

1. Stage appropriate files with `git add`
2. Draft commit message following conventions above
3. **Request user review of the commit message before committing**
4. Execute commit after approval

## Examples

```
feat: Add support for custom labels
fix(parser): Handle empty issue list gracefully
docs: Update CLI usage examples
chore: Update dependencies
```
