# Contributing to convocortex-stt

## Contributor License Agreement

Before your contribution can be accepted, you must agree to the
[Individual Contributor License Agreement](CLA.md).

The CLA grants Us the right to distribute your contribution under the project's
open-source license (AGPL-3.0) and, if needed, under a commercial license. You
retain full ownership of your contribution and may use it however you like.

**How to sign:** Include the following statement in your Pull Request description
or a comment:

> I have read the CLA and agree to the terms of the convocortex-stt Individual
> Contributor License Agreement.

Pull Requests without this statement will not be merged.

---

## What to contribute

- Bug fixes
- Platform compatibility (Linux, macOS)
- Documentation improvements
- Performance improvements to the audio pipeline or transcription
- New output handlers

If you plan a significant change, open an issue first to discuss the approach.

## Code style

- Follow the existing style in `stt.py` — flat functions inside `main()`, minimal
  abstraction, no unnecessary classes
- Keep it cross-platform where possible (`pywin32` stays optional)
- Do not add dependencies without discussion

## Running

```bash
uv sync --extra windows --extra nats
uv run python stt.py
```

## License

By contributing, you agree that your contributions will be licensed under
AGPL-3.0 (and potentially a commercial license per the CLA).
