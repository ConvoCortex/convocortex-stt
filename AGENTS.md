# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python application centered around `stt.py`, which runs the speech-to-text pipeline and runtime loop.
- `stt.py`: main entrypoint, model/audio runtime, event dispatch, gate control.
- `handlers.py`: local output handlers (file, clipboard, typing at cursor).
- `config.py` + `config.toml`: configuration loading and runtime settings.
- `state.py`: persisted runtime state across restarts.
- `README.md`: setup, runtime behavior, and integration notes.
- `CONTRIBUTING.md` / `CLA.md`: contribution and licensing requirements.

Keep new modules focused and flat; prefer extending existing files unless separation clearly improves maintainability.

## Build, Test, and Development Commands
Use `uv` for environment and execution.
- `uv sync`: install base dependencies.
- `uv sync --extra windows`: install Windows-only extras (clipboard support).
- `uv sync --extra nats`: install NATS integration extras.
- `uv run python stt.py`: run the engine locally.

Example full setup:
```bash
uv sync --extra windows --extra nats
uv run python stt.py
```

## Coding Style & Naming Conventions
- Python 3.10+.
- Follow existing style in `stt.py`: straightforward functions, minimal abstraction, avoid unnecessary classes.
- Use `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for module-level constants.
- Keep cross-platform behavior in mind; platform-specific features must remain optional.
- Do not introduce new dependencies without prior discussion.

## Testing Guidelines
There is currently no formal automated test suite in this repo. Validate changes by:
- Running `uv run python stt.py`.
- Exercising the modified path (audio flow, handler behavior, config toggles).
- Verifying no regressions in startup, transcription output, and optional integrations.

If you add tests, use `pytest`, place them under `tests/`, and name files `test_*.py`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit summaries (often snapshot-style), e.g. `snapshot prior to change: stt.py`. Keep messages concise and specific to touched files.

For pull requests:
- Explain what changed and why.
- Link related issues (if any).
- Include manual verification steps and platform used (Windows/Linux/macOS).
- Include the required CLA statement from `CONTRIBUTING.md`; PRs without it are not merged.
