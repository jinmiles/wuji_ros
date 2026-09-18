# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository. Read `README.md`, `docs/ARCHITECTURE.md`, and anything else under `docs/` that the task touches before non-trivial work.

## Execution Policy

**Default: hand over the command, don't run it.** Read code, inspect sample outputs, edit files, then write the exact command and stop.

- Applies to anything that executes the project: entrypoints, training, inference, optimization, evaluation, smoke tests, environment changes, and anything writing to data/output/checkpoint directories.
- Give one copy-pasteable block with env activation, all arguments, and log/output paths filled in. No placeholders — if a value is unknown, ask. Say what it should produce and what to look for in the result.
- Run directly without asking: read-only inspection (`git status --short`, `git diff`, `rg`, file/dir reads), `python -m py_compile`, and `--help` on a changed CLI.
- Run a project command yourself only when the user asks in that turn, and print the command before running it. Permission from an earlier turn doesn't carry to a new heavy job.

## Parallel Runs

**Default: fan out across GPUs, not over time.** Same setting over several inputs (sequences, configs, seeds) with free GPUs → one input per GPU in a single block, never a sequential chain.

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i <entrypoint> --input "${inputs[$i]}" \
    --out <out_root>/<run_id>/"${inputs[$i]}" \
    > <log_root>/<run_id>/"${inputs[$i]}".log 2>&1 &
done
wait
```

- Separate log file and output directory per process; never two processes writing the same checkpoint, result, or temp file. Start all, then `wait`, then collect and compare together.
- One sweep at a time: don't stack unrelated heavy jobs on a GPU, don't start a second sweep while the first runs.
- If a run doesn't fit on one GPU or inputs outnumber free GPUs, don't silently go sequential — report the cause and current GPU/memory usage, then propose a batching plan.

## Environment

Use the environment the repo already documents (conda env, venv, lockfile); don't create a new one and don't include `pip install` as routine setup. Keep per-stage environments separate.

```bash
conda activate <project-env>
conda run --no-capture-output -n <project-env> python <entrypoint> ...   # one-shot
```

## Boundaries

- Keep project behavior in project code. Never edit vendored/upstream code (`third_party/`, submodules, site-packages) to get project-specific behavior — integrate through wrappers, `sys.path` injection, and path/config overrides the project owns. Patch upstream only on explicit request.
- Data, cache, output, and scratch directories are git-ignored workspace. Never commit generated data, model assets, videos, checkpoints, plots, or analysis outputs.
- Keep report wording precise; don't claim data quality, accuracy, or validity the runs don't support. Health- or safety-adjacent output is not a diagnosis or clinical claim.
- Leave backups (`docs/experiments/_backup/` and any the user keeps) untouched unless the user asks about that backup specifically. Cleaning up docs is not permission to delete them.

## Contracts and Invariants

Confirm from code and a real sample output — never guess — at every boundary a change touches: paths, keys and shapes, frame/index ordering, side conventions, coordinate frame and units, transform direction and order, vertex or temporal correspondence.

- Preserve mesh topology and vertex IDs, side labels, frame IDs, and temporal correspondence across stages.
- Never re-match with nearest-neighbor or ICP where a correspondence already exists.
- Apply each alignment or transform exactly once; name source and target frame explicitly at geometry boundaries.
- Ground truth is for offline validation and evaluation only — never a runtime solver input, model input, or per-frame candidate selection.
- Use the default path documented in `docs/ARCHITECTURE.md`; an alternative backend or solver becomes the main path only on explicit request.

## Architecture and Style

- Core logic in the Python package; shell only activates the environment and forwards arguments; CLI/handler layers stay thin.
- One source of truth for repo paths and output layout — derive output locations from it, never hardcode at call sites. Keep layout deterministic and mirroring the input layout.
- Explicit stage boundaries: defined artifacts in, defined artifacts out. Reuse cached stage outputs on re-run with an explicit `--force`-style flag to regenerate.
- Load large NPZ/geometry once per process or cache it; never inside a loop.
- New options must be parallel-safe: no shared mutable state, no fixed output path, run identity derived from arguments.
- Straightforward Python, type hints on public signatures, `pathlib.Path` everywhere, structured parsers instead of ad hoc string parsing.
- Implement the needed algorithm in full but keep the diff minimal. No unnecessary abstraction, path guards, broad exception handling, or verbose code. Keep existing CLI flags and file formats unless the change requires otherwise, and keep defaults aligned with `README.md`.
- Use the project's logger (`get_logger()` / `configure_logging()` or equivalent), not `print()`. Log only at boundaries that confirm a shape, frame, transform, or runtime assumption; never dump large arrays, full payloads, weights, or subject data.
- Comments only for non-obvious math, coordinate-system choices, or upstream-integration details. No emoji in comments.
- Avoid hidden global state; keep configuration hooks in one documented place.

## Experiments

- No one-off files per experiment (`E###`). Add a small option or function to the existing trainer/inferencer/evaluator and record arguments and results in the topic's `experiments.md` and the artifact manifest. Don't duplicate an existing runner — a new file needs independently reusable logic and a stated reason in the report.
- Tests guard production behavior and data/geometry contracts, not one experiment's hard-coded paths, splits, or numbers. Remove temporary launchers, evaluators, and tests when the experiment ends; only adopted shared behavior stays.
- Resuming a goal: read the current goal, the user's latest instructions, `plan.md`, the last verdict in the topic's `experiments.md`, and `docs/TODO.md`. Map plan milestones to actual experiment numbers and flag missing items.
- Before handing over a training command, fill the artifact manifest: plan version/hash, hypothesis and why, parent run/hash, the single change, control conditions, data and init history, budget, metrics and non-inferiority criteria, falsification condition, exact command, contract check. No required field left empty.
- One change per experiment; parallel fan-out is for one change over several inputs, not several hypotheses. After results, record actual numbers and regressions, cause evidence, limitations, adopt/refine/reject, next hypothesis, and remaining tracked items.
- After two uninformative failures of the same hypothesis, don't propose a rerun with only weights or epochs changed — re-analyze current best, failure range, cause, and next hypothesis, and document any change of plan before proposing the next run.
- Never lower completion criteria to fit results, and never record an unrun configuration as failed; an improved component stays a candidate. Don't mark a goal complete until every completion condition in `plan.md` is met.
- For server-side benchmarks, prepare submissions only for candidates already passing local completion and non-regression criteria, following the repo's leaderboard doc for submission, quota, and collection.

## Docs and TODO

- Track unresolved multi-step problems in `docs/TODO.md` with baseline, cause hypothesis, and next validation. Record the verdict there; nothing is done before the evidence is confirmed. Don't touch items the user put on hold. On completion, move a short problem/solution note plus timestamp to `docs/TODO_completed.md` and delete the `docs/TODO.md` entry.
- Experiment and report Markdown written or updated in the current task is in Korean; code symbols, CLI flags, metric/key names, artifact paths, and raw logs stay English. Don't bulk-translate unrelated existing docs.
- Top-level numbers in `docs/experiments/README.md` are major topics, not ablations. A new topic gets `docs/experiments/{topic_idx:03d}_{topic_name}/experiments.md` and one row in the root README; ablations sharing that goal accumulate there as `E001`, `E002`, ... Never delete, reuse, or renumber failed or aborted numbers, and don't create per-ablation READMEs or subfolders.
- One row per experiment: `상태 / 바꾼 것 / 핵심 결과 / 결론`, plus only the conditions and artifacts the current experiment needs as a short section in the same file.
- Don't rename legacy `docs/experiments/<legacy_name>` directories or add new results there without an explicit migration request.
- Summaries name the actual input, change, and conclusion in plain sentences, not an experiment alias.
- Update `README.md` / `docs/` when CLI behavior, outputs, setup, or analysis semantics change.

## Validation

Smallest check that actually exercises the change, preferring what the repo already provides (test suite, smoke script, `--help`). Syntax and import checks first, then a smoke test on the smallest representative input.

```bash
python -m py_compile <changed files>   # run directly
python <entrypoint> --help             # run directly
bash -i scripts/shells/<name>.sh ...   # hand over
```

- Prefer CPU-only / synthetic-input smoke runs unless the change touches the heavy path. Never propose full-sequence inference or long optimization unasked.
- Geometry changes: scale, handedness, transform order, alignment. Temporal changes: frame IDs and correspondence. Runtime changes: no repeated loading of large data.
- Ad hoc test code and outputs go in the repo's git-ignored scratch directory; include the cleanup command when the check is done.

## Reporting

Conclusion and evidence first, then: files changed, behavior changed, checks run locally, commands handed over and still pending, validation not performed (full-sequence, GPU, missing data), assumptions and coordinate conventions confirmed, reason for any new file, and the added `E###` row (root README only if a new topic). Running an experiment is not finishing the research — answer the user's concern, don't just report that a run happened.

## Git and Commits

- No commits without user confirmation. Review `git status --short` and stage only intended changes — never unrelated user work.
- English messages, lowercase Blockly convention `<type>: <description>` under 256 chars. Types: `chore`, `deprecate`, `feat`, `fix`, `release`; append `!` for breaking changes.
- No `Co-Authored-By: Claude` or other AI attribution trailer. No emoji.
