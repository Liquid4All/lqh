# LQH persistency: current state and implementation plan

Date: 2026-07-17 (revised same day after design review)

## Executive summary

LQH already has good low-level durability for files and long-running jobs. It can preserve a conversation, rediscover local runs after a restart, reconnect to SSH and cloud jobs, replay missed cloud events, download completed cloud data-generation output, and retain cloud artifacts plus partial lineage. This is substantially more than the `remote_jobs.json` mechanism anticipated in `PERSISTENCY.md`.

The main gap is at the project level: after a session ends (or is compacted away), there is no durable record of what the user was trying to achieve, which decisions were made, and how the artifacts on disk relate to each other and to the spec that produced them.

An earlier draft of this plan answered that gap with a reconciled entity index, a structured handoff schema, a 10-step startup reconciliation sequence, and several new mechanisms for the agent to interact with project history. That draft was rejected in review for being over-engineered: **the consumer of project state is an agent that can gather context proactively** with the read tools it already has (`read_file`, `list_files`, `summary`, `artifacts`, `training_status`, `remote_status`, `list_deployments`, `load_skill`). Pre-computing a world-view for a reader that can look for itself is wasted machinery that will drift from reality exactly where it matters.

The revised plan follows one principle:

> **Engineer the facts, not the understanding.**

For every mechanism, apply the deletion test: *if this file didn't exist, could a competent agent with read tools reconstruct the information?*

- **No** → it is a durable fact. Engineer it well: append-only transcripts, provenance recorded at artifact-creation time, stable project identity, no-overwrite guards. No amount of proactive tool-calling recovers these later.
- **Yes** → it is a derived view. Do not build it. Let the agent read the facts and do the reasoning; that is what the model is good at and what code is bad at.

The one thing pure pull cannot do is discover **unknown unknowns** — a cloud job that finished while LQH was closed, an orphaned submit intent with billing implications, a spec edited outside the tool. Startup therefore injects a short list of **attention signals** ("2 jobs running, 1 submit-intent with unknown fate, spec changed since dataset X was generated — investigate with your tools"), not a dossier.

The highest-priority work is unchanged in substance:

1. Stop context compaction from destroying the only full conversation history; make session persistence atomic and resumable.
2. Give the agent an honest startup signal line and a truthful `summary` tool, plus an agent-maintained prose notes file for judgment that cannot be derived from disk.
3. Use a stable project identity for cloud state instead of the directory basename.
4. Record provenance in small co-located manifests at artifact finalization, and refuse accidental overwrite of expensive outputs.

## Tool policy (added in revision)

**No new agent tools.** Everything the agent needs is reachable through tools it already has:

- Project history, manifests, and notes are ordinary files → `read_file` / `list_files`.
- Cloud facts already have tools (`artifacts`, `training_status`, `remote_status`, `list_deployments`) or are folded into the existing `summary` tool.
- The agent's own handoff notes are written with the existing file-editing tools, guided by the system prompt and skills — not by a dedicated tool.
- Production failure cases are ordinary files the user drops into the project; the agent reads them and builds supplemental datasets with the existing pipeline tools. Skill guidance, not tooling.

The only tool-layer changes permitted by this plan:

1. **`summary` output changes** — fix its honesty gaps and add the signal line (see "Startup and resume behavior").
2. **Write-path side effects** — existing dataset/eval/training handlers write a manifest on finalization and refuse to overwrite an existing logical output. These change handler behavior, not the tool surface.

If a future scenario genuinely requires a new tool, that is a plan change to be argued explicitly, with the deletion test as the bar.

## Scope and source-of-truth model

The directory remains the project. No database is required to understand a local project, and LQH continues to work offline using the last locally known cloud state.

Source precedence:

1. **User-visible files and immutable run outputs**: `SPEC.md`, pipeline code, prompts, scorers, dataset/run manifests, configs, results, checkpoints.
2. **Authoritative remote facts**: cloud job status, artifact registry, lineage, deployment state, spend.
3. **Append-only history**: raw conversation records and the best-effort project log.
4. **Agent-authored notes**: judgment and intent that cannot be derived from disk — advisory prose, never treated as ground truth for job or artifact state.

There is deliberately **no derived-cache layer** (no reconciled index). A cached cloud snapshot for offline startup is the single exception, and it must always be safe to delete.

## What exists today

### 1. Plain-file project artifacts

The documented project layout already gives LQH durable, inspectable artifacts:

- `SPEC.md` and `other_specs/`;
- `data_gen/*.py`;
- `datasets/*/data.parquet` and co-located scores/summaries;
- `prompts/` and `evals/`;
- `runs/*/config.json`, progress, logs, metrics, and checkpoints.

Run and eval names are generally required to be unique. Training configs preserve the selected dataset paths, model, scorer, prompt, and hyperparameters. Training also supports multiple dataset batches, including repeats, so old expensive data can already be reused instead of discarded.

Git compatibility is documented, but LQH itself does not create commits or branches, require a clean tree, or record a complete local Git revision/dirty-state fingerprint for every artifact.

### 2. Conversation persistence and `/resume`

`lqh/session.py` stores conversations under `.lqh/conversations/<uuid>.jsonl`. The header records creation time, a preview, and token counts. Messages are saved after every append. `/resume` lists the ten newest conversations and reloads the selected message history. `/clear` starts another conversation.

This provides manual exact-session resume for ordinary conversations, but has important limitations:

- every save rewrites the whole file directly; it is not atomic and a kill during the write can truncate the session;
- there is no file lock or corrupt-tail recovery;
- metadata has no `updated_at`, active/interrupted/completed status, title, or last durable sequence;
- startup always creates a new conversation; it does not offer or default to the last interrupted one;
- resuming an old conversation does not refresh `SPEC.md`, the project summary, activity log, or cloud state, so the resumed context can be stale;
- `/clear` creates a new `Agent` but does not run `prepare_context()`, so it does not immediately inject the current spec and project state;
- there are no direct unit tests for `Session.save`, `Session.load`, `Session.list_sessions`, atomicity, corruption, or `/resume` selection behavior.

Secrets deliberately delivered out-of-band do not enter the transcript, which is a good existing privacy boundary.

### 3. Context compaction

`Agent._compact_context()` generates a summary from only the last approximately 20 stored messages, retains system messages and the last four messages, and replaces `session.messages` with that compacted context. The replacement is subsequently saved as the conversation file.

Consequences:

- conversation content older than the last 20 messages can disappear without ever being summarized;
- the persisted raw transcript is destroyed by compaction, so `/resume` cannot recover it;
- the summary and last four messages may overlap;
- compaction failures are silently ignored;
- there is no summary coverage marker such as “messages 1–230,” and no way to incrementally re-summarize from the last checkpoint.

This is the largest gap for Scenario 2, and it is the one place where “the agent can gather context proactively” is simply false: deleted history cannot be re-read by any tool. Compaction must become a derived API-context view over an intact transcript.

### 4. Startup reconstruction and the `summary` tool

For a new CLI session with an existing `SPEC.md`, `Agent.prepare_context()` injects:

- the full current `SPEC.md`;
- `handle_summary()` output;
- the last 50 entries from `.lqh/project.log`.

The summary scans specs, pipeline filenames, datasets, scorers, eval runs, training run names, and recent conversations. Dataset row counts and average scores are useful and cheap to obtain from Parquet metadata.

The summary is only a shallow inventory:

- the tool description promises prompts, but `handle_summary()` does not scan `prompts/`;
- runs are shown as unsorted names without config, local/remote status, progress, input datasets, metrics, checkpoint, or failure reason;
- it does not show active jobs, unknown submit intents, deployments, cloud artifacts, lineage, HF mappings, compute target, or spec drift;
- it cannot tell whether a dataset is a supplement or what produced it, except through filenames;
- caps such as 10 runs and 15 datasets can hide relevant state without saying what was omitted.

There are no direct unit tests for `handle_summary()` or `prepare_context()` reconstruction behavior.

### 5. Project activity log

`.lqh/project.log` is append-only JSONL. It records many useful events, including spec edits made through agent file tools, data-generation submission/completion/failure, scoring, training start/stop/completion/failure, and HF eval submission.

The log is a helpful recovery hint, not a reliable ledger: `append_event()` swallows every exception, writes are unlocked, not every operation emits an event, and manual edits outside the agent are absent. It should stay a best-effort hint; nothing may depend on every event having been written.

### 6. Local and remote job durability

This is the strongest area.

- Local subprocesses detach from the TUI and persist `pid`, `config.json`, `progress.jsonl`, stdout, and stderr. `SubprocessManager` is intentionally stateless and reconstructs status from those files.
- SSH jobs persist `remote_job.json`, and the TUI derives the remote from that file rather than from conversation state.
- Cloud jobs persist `remote_job.json` plus atomic `cloud_state.json`, including the last server event sequence. On reconnect, missed events are replayed without duplicating progress.
- Cloud submission uses an idempotency key written to `submit_intent.json` before the POST. A lost response therefore leaves a durable “fate unknown” marker rather than encouraging a duplicate billable submission.
- If accepted cloud state cannot be persisted locally, the client attempts to cancel the accepted job.
- The TUI scans run directories after restart, reattaches watchers, syncs progress, and records terminal transitions.
- Cloud data generation has a durable finalization marker, retry/backoff for artifact download, missed-event fallback through artifact lookup, and guards against an older cloud job overwriting newer/local data.

Tests cover cloud state persistence, response-loss idempotency, disconnect/replay, TUI job scanning, and completion after a CLI restart. These mechanisms are reused, not replaced.

Remaining gaps: orphaned `submit_intent.json` files are not automatically resolved, backend jobs with no surviving local run directory are invisible, and work that reached a terminal state while LQH was closed is not surfaced at startup.

### 7. Cloud projects, snapshots, artifacts, and lineage

The backend has first-class project support:

- a `projects` row is upserted on cloud submit;
- it caches display name, latest submitted spec SHA-256, base model, and reward model;
- `GET /v1/projects/{pid}` returns recent jobs, lifetime cloud spend, and a selected best checkpoint;
- `GET /v1/projects/{pid}/lineage` returns artifact lineage;
- the artifact registry retains cloud outputs, retention/pin state, and job association;
- checkpoint/eval publishers can record base model, parent artifact IDs, dataset artifact, hyperparameters, proxy/real metrics, image, code SHA, and job ID.

The Python client already defines `fetch_snapshot()`, `fetch_lineage()`, and `write_local_snapshot()` in `lqh/project_meta.py`. They are not called by the TUI or the summary path.

Cloud reconstruction is also incomplete at the data-model level:

- the project key used by cloud jobs and artifact tools is `project_dir.name`, while `.lqh/project.json` separately stores a stable UUID used only for telemetry;
- renaming a directory changes its cloud project, and two unrelated same-named directories for one user share a cloud namespace;
- copying a directory has no explicit continue-versus-fork semantic;
- the backend only stores the latest submitted spec hash, not a revision history, and the client never compares it with the current local spec;
- deployments are not part of the project snapshot.

### 8. Other persisted project settings

LQH also persists useful per-project state under `.lqh/`: permissions and cloud-spend consent, compute target and remote bindings, data-generation validation records bound to pipeline and source-content hashes, Hugging Face mappings, cloud dataset source sidecars, and a stable telemetry UUID.

The data-generation validation record is a good model for hash-bound derived state: editing the pipeline or its recorded inputs invalidates cloud submission until it is validated again.

## Scenario assessment

| Scenario | Current support | Assessment |
|---|---|---|
| 1. Interrupted work resumes later | Durable files, manual `/resume`, detached jobs, restart scanning | **Partial.** Exact resume exists but is manual, non-atomic, and stale w.r.t. current files/cloud state. |
| 2. Complex work spans context windows | Destructive compaction plus shallow new-session summary | **Weak.** Compaction permanently discards unsummarized history; startup lacks decisions and status. |
| 3. Compare multiple approaches | Git-compatible files, unique run names, immutable configs, sweeps | **Partial.** Works through naming conventions; nothing records which approach won and why. |
| 4. Return with deployed-model failure cases | `get_eval_failures`, multi-source training | **Partial.** External production cases have no ingestion convention linking them to a deployment/spec. |
| 5. Return with changed specifications | Editable `SPEC.md`, cloud stores latest spec hash | **Weak.** No artifact records which spec revision produced it, so nobody can explain what is still valid. |

The expensive-data requirement is only partly met: local generation and filtering can write into an existing output directory without a no-overwrite guard. Preservation currently depends on the agent choosing a fresh name.

## Design requirements

### R1. Preserve raw history; compact only a derived view

The raw transcript is append-only and is never replaced by a model-generated summary. API context is reconstructed as:

`system instructions + current spec/signals + latest durable summary checkpoint + uncovered transcript tail`

A checkpoint records which message sequence it covers. If checkpoint generation fails, the raw transcript remains intact.

### R2. Facts on disk, reasoning in the agent

Durable facts (manifests, configs, run state, cloud metadata) live in files and services the agent can read. Interpretation of those facts — what is stale, which approach is winning, what to do next — is the agent's job at read time, optionally recorded as prose notes. No component of LQH computes or caches a semantic world-view.

### R3. Push signals, not dossiers

Startup injection is bounded and consists of: the spec, the agent's notes, and a short signal list of things the agent would not know to look for (running/finished-while-away jobs, orphan submit intents, spec drift relative to manifests, offline/stale cloud cache). Everything else is pull.

### R4. Stable identity must survive rename and detect forks

Cloud project identity must not be the folder basename. A stored stable ID is used for jobs, artifacts, snapshots, lineage, deployments, and telemetry; the basename is only a display name. Copying a project results in an explicit choice: **continue** (retain identity) or **fork** (new identity with `forked_from` provenance).

### R5. Expensive outputs are immutable by default

Dataset, eval, and run creation fails if the logical output already exists, unless the caller explicitly allocates a new version or the user confirms an overwrite. A failed/retried run resumes inside its own run directory.

### R6. Spec drift marks provenance, not validity

Every finalized artifact records the spec hash it was built against. A changed spec makes dependent artifacts "built against an older spec" — a fact the summary can flag — never automatically invalid. Reuse, supplement, re-score, or regenerate is the agent's and user's call.

### R7. Local-first privacy

Conversations, notes, production failure examples, and user files stay local by default. Cloud synchronization is explicit and scoped. Cached cloud metadata must not contain secrets or signed URLs.

## Proposed on-disk model

Four things, three of which already exist in some form. No index file, no handoff schema.

### `.lqh/project.json`: identity

Extend the existing file:

```json
{
  "schema_version": 2,
  "project_id": "stable-uuid",
  "display_name": "support-triage",
  "cloud_project_id": "stable-backend-id",
  "forked_from": null
}
```

Identity creation must not be conditional on telemetry or authentication. Telemetry fields stay in a namespaced sub-object.

### `.lqh/conversations/<session-id>/`: immutable transcript

```text
meta.json          # schema version, timestamps, state (active|interrupted|completed), title, last seq
messages.jsonl     # append-only, one message per line
checkpoints.jsonl  # derived summaries with coverage ("covers messages 1–230"); safe to delete
```

Append under a lock; tolerate and quarantine a partial final line; replace `meta.json` and checkpoints via temp-file + `os.replace`. Legacy single-file sessions migrate lazily on load, keeping the original as a read-only backup.

### `NOTES.md`: agent-authored handoff (prose, not schema)

A plain markdown file at the project root, maintained by the agent through its existing file tools: current objective, decisions and rationale, what is blocked, next steps, which approach is selected and why. The system prompt and skills instruct the agent to update it at meaningful boundaries (before finishing a work phase, when a decision is made, when a long job is launched) and to read it at session start.

Properties that make prose the right format here:

- the writer and reader are both the agent; freeform text is its native format;
- it degrades gracefully — a stale note is a hint the agent verifies against the filesystem, whereas a stale schema field is silently trusted by code;
- the user can read and edit it, and it is Git-trackable next to `SPEC.md`.

It is advisory only. Job status, artifact existence, and metrics always come from their authoritative sources.

### Co-located artifact manifests

When an existing handler finalizes a dataset, eval, or filtered output, it writes a small `manifest.json` next to the artifact:

- logical name, creation timestamp, purpose (smoke, inspection, validation, training, failures, imported);
- row count and content hash (computed once, at finalization);
- producing pipeline path + hash, and source dataset paths/hashes;
- spec hash at generation time;
- scorer and filter threshold, if applicable;
- producing run/job ID and cloud artifact ID, if any;
- parent dataset, if this is a supplement.

Run directories keep `config.json` as-is, extended with the spec hash and a finalized result summary. Multi-source training records the exact dataset composition and repeats it trained on.

Manifests exist to be `read_file`'d by the agent (and humans), not parsed by a reconciler. Missing manifests on legacy artifacts are acceptable; absence of provenance is shown as absence, never invented from filenames.

### `.lqh/project.log`: keep as a best-effort hint

Minimal hardening only: add a session ID field, take an append lock, and log (not swallow) write failures. No schema overhaul, no event IDs, no causal graph. Nothing may depend on this log's completeness.

## Startup and resume behavior

Startup does five cheap things:

1. Ensure/migrate stable project identity; mark a previously `active` conversation from a dead process as `interrupted`.
2. Run the existing run-directory scan and watcher reattachment (already implemented).
3. If authenticated, make one snapshot fetch (`fetch_snapshot`) with a short timeout; cache it sanitized for offline reopen. Offline → use the cache, labeled stale.
4. Inject into the new conversation: `SPEC.md`, `NOTES.md` if present, and a **signal line** assembled from the scan + snapshot:
   - jobs running, and jobs that reached a terminal state since the last session;
   - orphaned `submit_intent.json` files (fate unknown — billing-relevant);
   - current spec hash differs from the hash recorded in recent manifests / last cloud submit;
   - cloud snapshot stale or unavailable;
   - a closing instruction: *investigate with your tools before acting.*
5. If the most recent conversation is `interrupted`, offer to resume it, preselected. Resuming restores the stored messages **verbatim** — step 4's injection is not part of the conversation at all, but an ephemeral system-context prefix that is rebuilt on *every* open (fresh, `/clear`, or resume) and never persisted. Continuity never pretends the filesystem hasn't changed, and injections can never duplicate into the transcript. `/clear` runs the same context preparation as a fresh start.

That is the whole sequence. There is no reconciler module, no `.lqh/index.json`, and no prioritized multi-section project brief; the `summary` tool (below) is the pull-side counterpart the agent calls when it wants detail.

### `summary` tool overhaul (existing tool, better output)

- include `prompts/` as advertised;
- show run status semantically (running / completed / failed + one-line reason), sorted by recency, with input datasets and checkpoint presence from `config.json` and manifests;
- include deployments and cloud artifacts from the cached/fresh snapshot;
- surface manifest provenance where present (purpose, spec hash match/mismatch, parent dataset);
- when output is truncated, say what was omitted ("12 older runs not shown"), never silently cap.

## Supporting iterative approaches and feedback

All three scenarios are handled with conventions and skill guidance over the primitives above — no new state formats, no new tools.

**Approaches (Scenario 3).** Distinct approaches live in distinct dataset/run/eval names (and optionally Git branches, recorded in manifests when present). Which approaches exist, how they compare, and which one won is exactly the judgment `NOTES.md` is for. The agent has the eval results and manifests to re-derive or re-verify a comparison at any time; skills should remind it not to compare metrics produced under different scorers/spec revisions without saying so.

**Production failure cases (Scenario 4).** Convention: the user (or agent) drops failure examples as JSONL/CSV/Parquet files under `feedback/`. The agent reads them with existing tools, records their origin (which deployment/checkpoint, when) in a manifest when it converts them into a supplemental dataset, and follows the existing remediation path: spec revision if needed → targeted pipeline → supplemental dataset → regression eval → mixed old+new training → compare → deploy. Old datasets stay immutable; training references them alongside the supplement. `get_eval_failures` output should be saveable into the same convention. This is skill/documentation work plus the manifest support from Phase 4.

**Spec changes (Scenario 5).** Manifests record the spec hash at generation time (R6). The signal line and `summary` flag mismatches. What to reuse, supplement, or regenerate is reasoned by the agent from the spec diff (Git or the agent's own comparison) and written down in `NOTES.md`. No revision database, no dependency invalidation engine.

## Delivery plan

### Phase 0: characterization and safety tests

- Session round-trip, ordering, malformed header/line, partial final line, concurrent append, kill-during-write simulation.
- Compaction of a >20-message conversation proving all raw messages survive (currently fails — that's the point).
- Startup, `/clear`, and `/resume` proving spec + state are injected exactly once.
- `handle_summary()` coverage and truncation honesty.
- Snapshot helper tests: offline cache, 404, auth failure, drift comparison.

### Phase 1: durable conversations

1. Directory-per-conversation format with append-only `messages.jsonl`, locked appends, atomic `meta.json`; lazy migration of legacy files with backup.
2. Compaction becomes derived: write a coverage-aware checkpoint, build API context from checkpoint + tail, never mutate stored messages. Compaction failure leaves everything intact and is logged.
3. Interrupted-state tracking; offer the newest interrupted session at startup.
4. `/resume` and `/clear` refresh project context (spec + signals) as described above.
5. Introduce the `NOTES.md` convention in the system prompt and relevant skills.

Fixes Scenarios 1 and 2 with no backend changes and no new tools.

### Phase 2: honest signals and cloud wiring

1. `summary` overhaul (prompts, semantic run status, deployments, truncation honesty).
2. Startup signal line: running/terminal-while-away jobs, orphan submit intents, spec-hash drift, snapshot staleness.
3. Wire `fetch_snapshot`/`write_local_snapshot` into startup; fold snapshot facts into `summary`.
4. Minimal project-log hardening (session ID, lock, observable failures).

### Phase 3: stable identity (coordinated Python + backend)

1. Project identity independent of telemetry; used for every cloud project/artifact/deployment operation.
2. Migration/alias from existing `(user, directory-basename)` projects so current cloud history stays reachable.
3. Detect folder copies; explicit continue/fork choice recorded in `project.json`.
4. Include deployments and paginated jobs/artifacts in the snapshot; retain submitted spec hash per job/artifact rather than only the latest.

### Phase 4: immutable provenance

1. Manifests written at finalization for datasets, filtered outputs, and evals; run `config.json` extended with spec hash and result summary.
2. No-overwrite guard in local generation and filtering, with explicit version allocation (`_v2` style) and confirmed-overwrite escape hatch.
3. Multi-source training composition recorded in the run config.
4. `feedback/` convention documented in skills; `get_eval_failures` output saveable into it.

### Phase 5: future cloud sessions

Only after the local model is reliable: sync project identity, session metadata, `NOTES.md`, and explicitly selected transcript data into Modal Sandbox sessions; lease/ownership semantics so two active agents don't fight; manifests and metadata sync, not whole project trees; conflicts preserved on both sides.

## Acceptance criteria

1. Killing LQH during any message write leaves every previously acknowledged message readable.
2. Repeated compaction across a very long conversation never deletes raw transcript; resumed context contains a coverage-aware summary plus the unprocessed tail.
3. Resuming a month-old session also sees a spec edited yesterday and a cloud job that completed while LQH was closed — via the refreshed signal injection.
4. `/clear` starts a new conversation but retains the current spec, notes, and signals.
5. Renaming a project folder does not lose its cloud jobs/artifacts; two same-named folders do not collide.
6. Copying a folder asks continue-or-fork and records the choice.
7. Deleting `NOTES.md`, checkpoints, or the cached snapshot loses no artifacts or job state; the agent can rebuild its understanding from durable facts alone.
8. Reusing an existing dataset output name cannot silently destroy expensive data.
9. Every artifact finalized after Phase 4 can be traced to its spec hash, inputs, config, and producing run by reading its manifest.
10. A spec change marks affected artifacts as built-against-older-spec in `summary`/signals; nothing is auto-invalidated or deleted.
11. The agent tool surface is unchanged: no new tools were added for persistency; only `summary` output and write-path side effects changed.
12. Existing projects and legacy conversations migrate without changing their visible artifacts or losing access to basename-keyed cloud history.
13. Offline startup uses the last cached cloud snapshot, clearly labeled stale; reconnect refreshes it without duplicating jobs/events.

## Risks and mitigations

- **Stale or misleading notes:** `NOTES.md` is advisory by contract; system prompt instructs the agent to verify claims about job/artifact state against tools before acting on them.
- **Agent forgets to maintain notes:** acceptable degradation — the durable facts still exist; skills prompt updates at phase boundaries; not enforced by machinery.
- **Signal line misses something:** signals cover only unknown-unknowns; everything else is discoverable via `summary` and read tools, so a missed signal degrades to a pull the agent may make anyway.
- **Large-project scan cost:** manifests and Parquet metadata keep startup cheap; content hashes computed once at finalization, never rehashed at startup.
- **Concurrent CLIs:** append locks, atomic replacement, short-lived state-file locks only; never a project-wide lock during network or training work.
- **Migration ambiguity:** preserve legacy files, record migration version, basename-alias handling in Phase 3.
- **Partial lineage:** absence of a manifest is shown as absence. Never invent parents from filenames.

## What was cut from the earlier draft, and why

Recorded so the decision survives:

- **`.lqh/index.json` reconciled entity index** — fails the deletion test; everything in it is derivable by the agent from manifests and configs, and a semantic cache drifts from reality precisely in the edge cases that matter.
- **`.lqh/handoff.json` structured decisions/blockers/next-actions schema** — replaced by agent-authored `NOTES.md`; schematizing judgment is brittle, prose degrades gracefully, and the reader is a model.
- **10-step startup reconciliation sequence and prioritized project brief** — replaced by the 5-step signals startup; push only what the agent wouldn't know to pull.
- **`approach_id`/workstream identity threaded through sessions, events, manifests, jobs** — deferred indefinitely; naming conventions plus `NOTES.md` cover Scenario 3 until proven insufficient.
- **First-class comparison records** — same; a comparison is prose plus re-derivable eval facts.
- **Typed `project_events.jsonl` with event IDs, actors, causal parents, idempotency keys** — the log stays a best-effort hint with minimal hardening; nothing may depend on its completeness, so a rich schema buys nothing.
- **Failure-batch schema and importers** — replaced by the `feedback/` file convention plus manifest provenance; no ingestion tooling.
- **Spec revision database and dependency invalidation** — replaced by spec hashes in manifests plus agent reasoning over the diff.
- **Any new agent tools for interacting with project history** — see Tool policy; the read tools that exist are sufficient.
