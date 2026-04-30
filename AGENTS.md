# Repository Routing

This directory is the GitHub-connected application repo.

For Codex, Claude, Qwen Code, and other coding agents:

1. Run code changes, commits, pushes, pull requests, branch checks, and CI/debugging actions from this `pipeline/` directory.
2. Use `origin`, which points to `https://github.com/hafizna/ai-analisis-gangguan-tfa.git`.
3. Do not treat the parent folder as the GitHub application repo. The parent folder is a local workspace wrapper for data, experiments, and helper files.
4. If working from the parent workspace, `cd pipeline` before running GitHub-related commands.

The parent workspace may record this repo as a submodule pointer, but application history belongs here.
