# Repository Routing

This directory is the GitHub-connected application repo.

For Codex, Claude, Qwen Code, and other coding agents:

1. Run code changes, commits, pushes, pull requests, branch checks, and CI/debugging actions from this `base_ai_tfa/` directory.
2. Use `origin`, which points to `https://github.com/hafizna/base_ai_tfa.git`.
3. Do not treat the parent folder as the GitHub application repo. The parent folder is a local workspace wrapper for data, experiments, and helper files.
4. If working from the parent workspace, `cd base_ai_tfa` before running GitHub-related commands.
