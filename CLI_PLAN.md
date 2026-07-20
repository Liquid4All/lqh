# CLI mode for third-party agentic harnesses — design plan (rev 4)

Response to `CLI.md`. Grounded in the current code; file references
are to `lqh_py/lqh/...`. Rev 2 incorporates two feedback rounds
(2026-07-17): a simplification of the permission model, and a senior-IC
review that correctly flagged several places where rev 1 understated the
runtime separation needed. Changes from rev 1 are marked **[rev2]**;
rev 3 refined `--help` and the README section. **Rev 4 (2026-07-20)**
re-grounds the plan on v0.4.19 after the persistency layer
(`PERSISTENCY_PLAN.md` Phases 0–4) landed — ephemeral context injection,
stable project identity, copy detection, overwrite guards, manifests,
startup signals — and adds the `lqh hello` bootstrap command. Changes
are marked **[rev4]**.

## 0. TL;DR recommendation

Build **both** surfaces CLI.md asks for, as plain subcommands on the existing
`lqh` entry point:

```
lqh                          # TUI, unchanged (bare invocation)
lqh hello                    # harness bootstrap: print the agents guide  [rev4]
lqh run "<task prompt>"      # sub-agent mode: headless agent loop, one task
lqh tool list|schema|call    # direct tool access, JSON envelope on stdout
lqh docs agents|skill <n>    # capability docs for harnesses (hello ≡ docs agents)
lqh project continue|fork    # resolve a copied-folder identity headlessly [rev4]
lqh login [--no-browser]     # device-flow auth without entering the TUI
```

No separate "cli mode" flag — the subcommand *is* the mode. Bare `lqh` (and
`lqh --auto`) keep today's behavior.

**[rev2] Honest sizing:** the `lqh tool` surface is small. `lqh run` is a
**medium refactor**, not "productionize the E2E harness": it needs the
`auto_mode` boolean split into an explicit policy object (§4.2) and a
background-job supervisor extracted from the TUI (§4.5). The foundations
(structured tool results, tool metadata, argument validation) come first
(§8) so the public JSON contract doesn't churn after release.

## 1. What the code gives us (verified findings)

- **Agent core is headless-safe.** `Agent` (`agent.py:401`) imports no TUI
  modules; all integration goes through the optional `AgentCallbacks`
  bundle (`agent.py:328`). `tests/harness/harness.py` already drives the
  full loop without Textual.
- **Tools are a registry.** `TOOL_HANDLERS` (`handlers.py:5969`) →
  `execute_tool(name, args, project_dir, **extra)` (`handlers.py:6019`).
  Handlers return `ToolResult` sentinels interpreted by
  `Agent._handle_tool_call` (`agent.py:1096`).
- **[rev2] But the seams are rougher than rev 1 claimed:**
  - `auto_mode=True` bundles ≥5 independent policies: auto skill injection
    (`agent.py:462`), cloud-compute default persistence (`agent.py:1219`),
    project-wide permission auto-grant (`agent.py:1239`), secret→`.env`
    fallback (`agent.py:1295`), and the auto-only terminal tools
    (`definitions.py:1987`).
  - Consent is inconsistent per domain: script exec and cloud data-gen use
    out-of-band `_script_consent`/`_cloud_consent` params
    (`handlers.py:979`); training and HF push read
    `.lqh/permissions.json` directly inside the handler
    (`handlers.py:4104`, `:3384`); the script and HF `PERMISSION_REQUIRED`
    sentinels don't populate `permission_key`.
  - `ToolResult` (`handlers.py:45`) has **no** `ok`/error fields — failures
    are prose (`"Error: …"`, `"❌ …"`). A reliable exit-code contract needs
    structured results first.
  - Background-run supervision (job registry, completion queue,
    `on_await_background` parking, scoring watchers, cleanup) lives in
    `LqhApp`, not the agent. Without a supervisor a headless agent
    busy-polls and never receives completion events.
  - Session locks protect appends, not run ownership (`session.py:219`);
    `save_permissions` is an unlocked read-modify-write
    (`permissions.py:48`).
- Sessions are durable and resumable (`.lqh/conversations/<uuid>/`,
  `Session.load`/`list_sessions`/`repair_states`).
- **[rev4] The persistency layer changes the headless picture — mostly
  for the better:**
  - `prepare_context()` (`agent.py:1602`) is now the *ephemeral* context
    rebuild: SPEC.md + `NOTES.md` + a one-line artifact inventory + the
    attention-signal block (`signals.collect_signals`). It runs on every
    open (fresh, `/clear`, `/resume`), is never persisted into the
    transcript, and reads only local state (incl. the cached snapshot).
    A headless run therefore gets startup signals — jobs finished while
    away, orphan submit intents, spec drift — for free.
  - Stable identity and copy detection (`project_identity.ensure_identity`
    / `detect_copy`) are called **only** from `LqhApp.run`
    (`tui/app.py:3281`) today. A headless boot that skips them would run
    without identity and, worse, silently continue on an unresolved copy
    that TUI auto mode refuses to start on. Cloud keying
    (`cloud_project_key`) fails closed on a corrupt identity file.
  - The Phase-4 overwrite guard (`dataset_guard.py`) refuses reuse of an
    existing logical output; the escape hatch is the explicit
    `overwrite: true` **tool argument** (already in `definitions.py`),
    not an interactive prompt — headless-compatible as-is.
  - Manifests (`manifest.py`) are written at finalization and are plain
    files — exactly the provenance a third-party harness can `cat`.
- Startup cost is all in `LqhApp` (telemetry session, update check, login
  prompt, cloud snapshot, job scan). Minimal headless boot **[rev4]**:
  `ensure_identity` → `detect_copy` → `Session.repair_states` →
  `get_token` → `create_client` → `Session` → `Agent` →
  `prepare_context()` — local-only; the snapshot fetch is an optional
  extra for fresh (vs. cached) cloud signals.

## 2. Settled decisions

1. **Help surface:** `lqh --help` stays human (~10 lines) **plus** a 3–5
   line "For agent harnesses" section pointing at `lqh docs agents` and the
   key subcommands. Full machine docs behind `lqh docs agents`.
2. **Sub-agent shape:** free-form `lqh run "<task>"`, headless agent loop,
   auto-mode-style no-user rules, structured JSON exit. Stage presets are
   later sugar; `--auto` (TUI) remains the full-pipeline mode for humans.
3. **Tool output:** JSON envelope by default on stdout; `--pretty` for
   humans; exit code mirrors the envelope.
4. **[rev2] Permissions:** see §3 — rev 1's fail-fast-everywhere model is
   replaced by *invocation-is-consent* for direct calls and a two-tier
   policy for `lqh run`.

## 3. Permission model **[rev2 — rewritten for simplicity]**

### 3.1 Why the store exists, and why that matters here

`.lqh/permissions.json` exists to gate **lqh's own autonomous model** from
acting without the human. It is not a security boundary against the
*caller*: whoever invokes `lqh tool call` already holds an interactive
shell (can run `python data_gen/x.py` directly) and the API token (can
spend). Gating direct calls adds ceremony without adding safety — and the
distinction between "executes Python" (`run_data_gen_pipeline`) and
"merely costs money" (`start_training`) collapses, because the calling
harness has both capabilities anyway and applies its own consent layer
(Claude Code prompts its user before running any command).

### 3.2 Direct tool calls: invocation is consent

`lqh tool call` **bypasses the permission store entirely** — the consent
question was already answered by the human driving (or configuring) the
harness. Implementation: the CLI supplies full consent through the same
`PermissionContext` used by run mode (§3.4). No `--allow` flags, no grant
hints, no permission exit code on this surface. Document it in
`lqh docs agents` in one line: *"direct tool calls are pre-consented; your
harness's own permission system is the gate."*

This deletes rev 1's `lqh grant` command, per-invocation grant threading,
and exit code 3 from the tool surface — the complexity Mathias flagged.

**[rev4]** Invocation-is-consent bypasses the *permission store*, not the
Phase-4 overwrite guard: that guard is arg-gated (`overwrite: true`), so
expensive outputs stay protected on this surface too. An overwrite
refusal surfaces as `error_kind: "conflict"` (§5.3) with the version-
allocation hint (`_v2`) in the message.

### 3.3 `lqh run`: gate publishing, not work

The sub-agent *is* an autonomous model, so policy is warranted — but the
delegated task already implies doing the work. Two tiers:

- **Auto-granted** (task-implied, project-scoped): script execution,
  training, cloud data-gen, evals. Same stance as TUI `--auto`. Spend
  safety is the backend's job — org/user monthly cost limits are enforced
  server-side pre-flight; a consent prompt is the wrong tool for budget
  control. (Future, not v1: a client-side `--max-cost` advisory.)
- **Gated by default** (outward-facing publishing): `hf_push`,
  `push_to_production`, `create_inference_key`. Enable with
  `lqh run --allow-publish` or a prior durable grant (made in the TUI).
  An ungated hit terminates the run with `status: "needs_permission"` +
  the exact re-invocation, and the session id for resume (§4.4).

One flag, one rule ("code + compute auto; publishing gated"). A
`--strict` flag can later gate the first tier too for cautious users; not
v1.

### 3.4 Mechanism: one `PermissionContext`

The IC review is right that consent paths are scattered (§1). Introduce a
small invocation-scoped `PermissionContext` (in `tools/permissions.py`)
consulted by **all four** check sites — replacing `_script_consent` /
`_cloud_consent` params and the in-handler store reads. Sources, in
precedence order: CLI surface policy (full consent for `tool call`;
tier rule for `run`) → durable store → deny with sentinel. All
`PERMISSION_REQUIRED` sentinels must then populate `permission_key`
(fixing the script/HF gaps). The TUI keeps its interactive prompting on
top of the same context, so behavior there is unchanged.

## 4. Sub-agent mode — `lqh run`

### 4.1 Invocation

```
lqh run "<task>" | --prompt-file f | -   # stdin; long prompts are normal [rev2]
        [--project DIR]                   # explicit, not cwd-only [rev2]
        [--resume <session-id>] [--allow-publish]
        [--max-turns N]                   # N = LLM calls [rev2: defined]
        [--max-tool-calls N]              # N = total across run [rev2]
        [--quiet]
```

Output: **JSON result on stdout, always** (no `--md`; a harness that wants
prose reads `.summary`) **[rev2]**. NDJSON progress events on stderr
(`tool_call`, `agent_message`, `stage`, `progress`), each with
`schema_version`, `run_id`, and a `seq` number **[rev2]**. Stray library
stdout is redirected to stderr at the fd level so the one-JSON-object
guarantee is real **[rev2]**.

### 4.2 `HeadlessPolicy` — split the `auto_mode` boolean **[rev2]**

Adopting the IC recommendation: before building `lqh run`, refactor
`Agent(auto_mode: bool)` into an explicit policy object (the boolean
becomes a preset, TUI behavior unchanged):

```python
@dataclass(frozen=True)
class AgentPolicy:
    no_user: bool                  # ask_user interception, continue-nudges
    sticky_skill: str | None      # "auto" | "subagent" | None
    permissions: PermissionContext # §3.4
    compute_default: str | None   # persist "cloud" when unset, or fail
    secret_delivery: str           # "prompt" | "env" | "result" | "discard"
    terminal_tools: bool           # expose set_auto_stage/exit_auto_mode
```

Presets: `TUI_INTERACTIVE`, `TUI_AUTO` (≡ today's `auto_mode=True`),
`SUBAGENT` (no_user, `subagent` skill, §3.3 permissions, compute_default
from config **or a `needs_configuration` exit — not the permission exit
code [rev2]**, secrets → result payload, terminal tools on). This removes
the fragile "auto mode with outside overrides" shape rev 1 implied.

Secrets: **no silent `.env` append** in headless mode; the key rides in
the result payload, `--save-secret` opts into persistence **[rev2]**.

### 4.3 The `subagent` sticky skill

New `lqh/skills/subagent/SKILL.md`: "you are a sub-agent of an external
harness; do the delegated task and only that task; never `ask_user`;
prefer conservative defaults; report via `exit_auto_mode` with the
structured fields (§4.6); publishing tools may be denied — treat that as a
terminal `needs_permission`, don't retry." Per-stage skills remain
loadable as usual.

**[rev4]** Two persistency-driven additions to the skill: (1) *maintain
`NOTES.md`* — a sub-agent is exactly the "agent finishing a work phase"
the persistency plan's handoff convention targets, and its notes are how
the next session (TUI or headless) picks up its decisions; update it
before exiting. (2) *Respect output immutability* — never pass
`overwrite: true` on its own judgment; allocate a `_v2`-style version
instead, unless the delegated task explicitly names an output to replace.

### 4.4 Resume is contextual, not positional **[rev2]**

Rev 1 claimed resume picks up "exactly where it stopped". Not true: a
permission-blocked tool call must receive a synthetic tool result before
exit (same repair as `abort_turn`, `agent.py:686`), and reloading a
session does not re-execute anything. `--resume <id>` therefore does
**contextual resume**: load the session, inject a message — "the
`publish` capability has been granted; retry the blocked action" (or the
user's new instruction) — and run a normal turn. Deterministic-retry
protocols are not worth the machinery; the model retrying with full
context is the design. Documented as such.

**[rev4]** The persistency layer strengthens this: `prepare_context()` is
rebuilt on *every* open and never persisted, so a resumed sub-agent
automatically sees the **current** spec, notes, and signals (including
jobs that finished between the two invocations) rather than the world as
it was when the session was interrupted. Contextual resume needs no extra
freshness machinery — it inherits it.

### 4.5 Background-job supervisor — extract from the TUI **[rev2]**

`training_status --wait` and any real `lqh run` need the supervision that
currently lives in `LqhApp`: the job registry, completion queue,
`on_await_background` parking, eval scoring/sync watchers, and
cancellation cleanup. Extract these into a headless `lqh/jobs.py`
supervisor (building on the existing `watcher.py` primitives) that both
`LqhApp` and the headless driver consume. Without this the sub-agent
busy-polls, misses completion events, and leaks tasks on cancel. This is
the single biggest work item in the plan and is scheduled before `lqh run`
(§8, phase 4).

### 4.6 Structured exit — grounded, not just claimed **[rev2]**

`exit_auto_mode` gains optional `summary`, `artifacts`, `metrics` fields —
but model output is a *claim*. Grounding:

- The headless driver keeps an **artifact ledger** populated
  deterministically from successful tool calls during the run (training →
  run dir + checkpoint, data-gen → dataset path, `hf_push` → repo id,
  `push_to_production` → deployment id/URL). This — not an mtime scan — is
  the authoritative artifact list; external IDs/URLs included, not only
  filesystem paths.
- Model-supplied artifact paths are validated (project-relative, exist)
  and merged; model metrics are labeled `"reported"` unless a scoring tool
  result corroborates them.

Result payload:

```json
{
  "schema_version": 1, "run_id": "…",
  "status": "success",   // success | failure | needs_permission |
                          // needs_configuration | auth_required |
                          // limit_exceeded | interrupted | timed_out  [rev2]
  "reason": "…", "summary": "…markdown…",
  "artifacts": [ {"kind": "checkpoint", "path": "runs/sft_v1", "source": "ledger"} ],
  "metrics": { "post_sft": {"value": 0.78, "provenance": "reported"} },
  "session_id": "…",
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "turns": 0 },
  "duration_s": 0
}
```

Exit codes: `0` success · `1` failure · `2` usage · `3` needs_permission ·
`4` auth_required · `5` needs_configuration · `6` interrupted/timed_out.

### 4.7 Telemetry **[rev2]**

Headless commands must not silently start telemetry: `lqh run` starts a
telemetry session **only if consent was previously recorded** (the TUI
notice); otherwise disabled, with a one-line stderr note. `lqh tool` is
always telemetry-free.

### 4.8 Persistency-layer contract for headless boot **[rev4]**

The identity/copy startup step is currently TUI-only (`tui/app.py:3281`).
Headless surfaces must honor the same invariants, so extract it into a
shared `headless_boot(project_dir)` helper both `LqhApp` and the CLI
call:

1. **`ensure_identity` first, unconditionally** — same R4 rule as the
   TUI: no command may run cloud operations in a project without a
   stable identity. Corrupt identity file → exit 5
   (`needs_configuration`) with the error; never silently replaced.
2. **`detect_copy` next.** An unresolved copy terminates `lqh run` and
   any *mutating/cloud* `lqh tool call` with exit 5 and the exact
   resolution commands (`lqh project continue` / `lqh project fork`).
   TUI auto mode already refuses to start on an unresolved copy; a
   headless agent silently continuing would be strictly worse (it could
   bill work into the original project's cloud namespace). Read-only
   local tools (`summary`, `list_*`) may proceed with a stderr warning.
   `lqh project continue|fork` is a thin wrapper over
   `record_continue_decision` / `fork_identity` so a pure-headless
   harness is never forced into the TUI to unblock itself.
3. **`Session.repair_states`** before creating the run's session, and —
   symmetrically — the headless driver must mark its own session
   `completed`/`interrupted` on exit. Sub-agent sessions live in the
   same `.lqh/conversations/` store, which is a feature: a `lqh run`
   session shows up in the TUI's `/resume` list (and vice versa via
   `--resume <id>`), and honest state keeps the TUI's
   interrupted-session offer and finished-while-away signals truthful.
4. **Signals ride along for free** — `prepare_context()` injects the
   signal block, so a delegated sub-agent starts each run knowing about
   orphan submit intents, terminal-while-away jobs, and spec drift, and
   the `subagent` skill's "investigate before acting" stance applies.
   `lqh run` does the one short-timeout `fetch_snapshot` refresh iff
   authenticated (same as TUI startup); `lqh tool` never does.
5. **`lqh status --json`** (phase 6) is now mostly free: it serializes
   `signals.collect_signals` + the run-directory scan instead of
   inventing a new aggregation.

## 5. Direct tool access — `lqh tool`

### 5.1 Commands

```
lqh tool list [--json]      # exposed tools: name, one-liner, classification
lqh tool schema <name>      # JSON schema from definitions.py
lqh tool call <name> --args '<json>' [--args-file f] [--pretty] [--wait]
```

`--args` is the same JSON object the orchestration model would emit —
`definitions.py` stays the single source of truth. Arguments are
**validated against the schema before dispatch** (small in-house validator
over `required`/`type`/`enum`; the schemas are simple) so malformed input
is a clean exit-2, not a handler traceback **[rev2]**.

### 5.2 Exposure: opt-in, tagged at the definition **[rev2]**

Reversing rev 1's default: `_tool()` gains metadata with
**`cli=False` by default** — a future tool is unexposed unless someone
decides otherwise. Metadata per tool: `cli`, `mutating` / `destructive`,
`needs_auth`, `permission_domain`, `needs_loop`. Private metadata is
stripped before definitions are sent to the API. A unit test asserts
`TOOL_HANDLERS` keys ≡ definition names (kills silent drift). A full
`ToolSpec` registry unifying schema+handler is deliberately deferred — the
metadata + parity test gives most of the value at a fraction of the churn.

Exposed set: **`summary`** (rev 1 wrongly excluded it — it is a read-only
project-state report, `handlers.py:669`, and the single most useful
discovery primitive for a harness), `list_user_data`, `list_models`,
`list_skills`, `get_eval_failures`, `run_data_gen_pipeline`,
`run_data_filter`, `run_scoring`, `start_training`, `training_status`,
`stop_training`, `start_local_eval`, `eval_hf_model`, `hf_push`,
`hf_pull`, `hf_repo_info`, `pull`, `push`, `artifacts`, `gguf_convert`,
`push_to_production`, `list_deployments`, `get_deployment`,
`stop_deployment`, `restart_deployment`, `create_inference_key`,
`list_inference_keys`, `revoke_inference_key`, `remote_*`, `compute_set`.

Not exposed: file tools (the harness has better ones), `ask_user`,
`load_skill` (→ `lqh docs skill`), `set_auto_stage`, `exit_auto_mode`.

### 5.3 Structured `ToolResult` before the contract ships **[rev2]**

The envelope's `ok`/exit-code rule cannot be built on prose sniffing. Add
backward-compatible fields to `ToolResult`:

```python
ok: bool | None = None          # None = legacy/unclassified
error_kind: str | None = None   # auth | permission | config | validation
                                 # | not_found | conflict | upstream | runtime
                                 #   [rev4] conflict = overwrite-guard refusal
retryable: bool = False
details: dict | None = None
```

plus a `ToolResult.fail(kind, msg, …)` helper. Sweep the **exposed**
handlers' error returns to use it (mechanical; `content` keeps the prose
for the agent). Unmigrated results fall back to `ok=None` → envelope
reports `"classified": false` and exits 1 only on the legacy `"Error:"`/
`"❌"` prefixes — explicitly best-effort until the sweep completes.

### 5.4 Envelope

```json
{
  "schema_version": 1, "ok": true, "tool": "start_training",
  "result": { "text": "…", "secret": null, "details": {} },
  "error": null,   // { "kind", "message", "retryable", "details" }
  "meta": { "duration_s": 3.2, "lqh_version": "0.4.18" }
}
```

Sentinel mapping (headless interpreter shared with run mode):
`SECRET_DELIVERY_REQUIRED` → secret into `result.secret` (+`--save-secret`
to persist); `COMPUTE_PICK_REQUIRED` → exit 5 with the exact
`compute_set` call as hint; `requires_user_input` → defensive exit 2
(tool not exposed); `PERMISSION_REQUIRED` → unreachable on this surface
(§3.2), defensive exit 3. `training_status --wait` parks on the extracted
supervisor (§4.5), not a poll loop.

### 5.5 Startup

`lqh tool` boots nothing from `LqhApp`: no telemetry, no update check, no
login prompt, no snapshot/job scan. Lazy imports keep `lqh tool list` in
tens of ms. Token read only when the tool needs the API; missing → exit 4
with `lqh login` hint. **[rev4]** The one thing it may *not* skip is the
identity contract (§4.8): any cloud-touching or mutating call runs
`headless_boot` first (cheap, local); `lqh tool list/schema` and
read-only local calls skip even that.

## 6. Docs exposure — `lqh docs`

```
lqh hello                  # harness bootstrap — alias of `lqh docs agents` [rev4]
lqh docs agents            # full harness-facing doc (markdown)
lqh docs skills            # list skills
lqh docs skill <name>      # print SKILL.md verbatim
```

**[rev4] `lqh hello`** is the memorable front door for third-party
harnesses: byte-identical output to `lqh docs agents` (one implementation,
two names). It exists so the *entire* onboarding instruction — in
README, `CLAUDE.md`/`AGENTS.md` stubs, blog posts, or a human telling
their harness what to do — collapses to a single self-explanatory
command: *"run `lqh hello` first."* It must work with zero project
state, no auth, and no network, in tens of ms (same lazy-boot rule as
`lqh tool list`).

`lqh docs agents` = checked-in template with the tool table generated from
`get_all_tools()` at print time; prints the package version. **It must be
fully self-contained** — a harness that runs it cold, with zero prior
context, learns everything it needs. Content order:

1. **What LQH is** — one paragraph: product intent, the "customize LFMs
   from a spec" premise, project-as-directory model (SPEC.md, `data_gen/`,
   `datasets/`, `runs/`, `.lqh/`).
2. The fine-tuning workflow (steps 1–12 from CLI.md) and where iteration
   re-enters it.
3. The two integration modes and when to use which: `lqh run` (delegate a
   task, get a JSON result + resumable session) vs `lqh tool …`
   (fine-grained single steps).
4. Consent model (§3, one line: direct calls are pre-consented; `run`
   gates publishing behind `--allow-publish`).
5. Contracts: envelope, exit codes, status set, NDJSON events, all with
   `schema_version`.
6. Worked examples (delegate data-gen; call `summary`; start + wait on a
   training run; resume after a grant).
7. **[rev4] Project conventions the harness must follow** — a harness
   with its own file tools plays the role `PERSISTENCY_PLAN.md` assigns
   to "the agent", so the guide teaches the same conventions the
   built-in agent gets from its system prompt and skills: read and
   maintain `NOTES.md` (advisory prose handoff; verify claims with
   tools); treat datasets/runs/evals as immutable — allocate `_v2`
   versions, `overwrite: true` only on explicit user intent; read
   co-located `manifest.json` files for provenance (spec hash, sources,
   producing run) instead of guessing from filenames; drop production
   failure cases under `feedback/`; heed the signal/`summary` warnings
   about spec drift and orphan submit intents before spending.

### 6.1 Proposed `lqh --help` **[rev3]**

Human section first and unchanged in spirit; `commands:` from the
subparsers; then the harness block as an epilog (~5 lines, decision #1):

```
usage: lqh [-h] [--version] [--auto SPEC_DIR] [--spec STRING] [command] …

Liquid Harness — agent for customizing Liquid AI foundation models into
task-specific models. Run `lqh` with no arguments in your project
directory to start the interactive TUI.

options:
  -h, --help       show this help message and exit
  --version        Show the installed lqh version and exit.
  --auto SPEC_DIR  Fully-autonomous mode: run the whole pipeline (rubric →
                   data gen → filter → baseline → SFT → DPO → report)
                   against SPEC_DIR/SPEC.md, no user prompts.
  --spec STRING    Extra sticky context appended to every agent turn.

commands:
  hello            Print the guide for AI agents driving lqh. Start here.
  login            Authenticate via device flow (works without the TUI).
  docs             Print docs: skills, and the agent-harness guide.
  run              Run one task headlessly; JSON result on stdout.
  tool             List / inspect / call individual pipeline tools (JSON).
  project          Resolve a copied project: continue or fork identity.

for AI agents and harnesses (Claude Code, Codex, …):
  If you are an AI agent driving lqh programmatically, first run:
      lqh hello
  It explains what LQH is, the full fine-tuning workflow, the headless
  commands (`lqh run`, `lqh tool …`), their JSON contracts, and the
  project conventions (NOTES.md, manifests, immutable outputs).
```

### 6.2 README section **[rev3]**

Phase 2 also updates `README.md` with a new section (after "🤖 Auto
mode"), e.g. **"🤝 Using lqh from Claude Code, Codex & other agent
harnesses"**:

- Two or three sentences: lqh's built-in agent is optional — any agentic
  harness can orchestrate the same pipeline via the headless CLI, either
  delegating whole tasks (`lqh run`) or calling individual steps
  (`lqh tool`).
- **The bootstrap command, front and center [rev4: `lqh hello`]**: tell
  the harness to run `lqh hello` before touching lqh. Show it both as a
  prompt —

  ```
  Run `lqh hello` to learn how to drive lqh, then create a spec for
  <my task> and generate a first draft dataset.
  ```

  — and as a standing instruction for the project's `CLAUDE.md` /
  `AGENTS.md`: *"Before working with lqh, run `lqh hello` and follow
  its contracts."*
- One short `lqh run` example with the JSON result, one `lqh tool call`
  example.
- A pointer that consent lives in the harness for direct calls, and that
  `lqh run --allow-publish` is needed for HF push / deployment.

Deferred: `lqh docs init` writing an `AGENTS.md` stub (the README snippet
covers the need manually for now).

## 7. Concurrency **[rev2 — was overstated]**

- **Exclusive run lock per session**: a headless run (and the TUI) takes
  an owner lock (pid + liveness, like `repair_states`) so two agent loops
  can't interleave one conversation.
- **Atomic permission writes**: `save_permissions` moves to
  lock-file + tmp-write + `os.replace`.
- Documented contract: concurrent **read-only** `lqh tool` calls alongside
  a live session are supported; concurrent **mutating** calls or two agent
  loops on one project are not (best-effort advisory lock + warning).

## 8. Implementation phases **[rev2 — reordered, foundations first]**

1. **Foundations** (no user-visible surface): structured `ToolResult` +
   `fail()` helper + sweep of exposed handlers; tool metadata + parity
   test + API-side metadata stripping; schema validator;
   `PermissionContext` unifying the four consent sites (TUI behavior
   unchanged); atomic permission writes.
2. **Read-only surface**: `cli.py` subparsers (explicit subcommands, no
   `parse_known_args`; regression tests that `lqh`, `lqh --auto`,
   `lqh --version` are byte-identical to today); `lqh tool list/schema` +
   `call` for read-only tools; `summary`; `lqh docs` (agents guide per
   §6, self-contained) **+ `lqh hello` alias [rev4]**; `lqh login
   [--no-browser]` with machine-readable output; `--help` per §6.1;
   README harness section per §6.2.
3. **Mutating tool calls**: full exposed set under invocation-is-consent;
   envelope + exit codes stable; `--save-secret`; `--wait` deferred to 4.
   **[rev4]** `headless_boot` extraction (§4.8: identity, copy check,
   `repair_states`) shared with the TUI; `lqh project continue|fork`;
   overwrite-guard refusals mapped to `error_kind: "conflict"`.
4. **Job supervisor**: extract registry/completion-queue/parking/cleanup
   from `LqhApp` into `lqh/jobs.py`; TUI consumes it; `training_status
   --wait` lands here.
5. **`lqh run`**: `AgentPolicy` split (presets, TUI unchanged); `subagent`
   skill (incl. NOTES.md + immutability guidance, §4.3); NDJSON events;
   artifact ledger + structured exit; `--allow-publish` tier;
   needs_permission termination. **[rev4]** Session state marked
   `completed`/`interrupted` on exit; optional snapshot refresh for
   fresh signals (§4.8).
6. **Hardening**: contextual resume message protocol, run locks under
   contention tests, limits (`--max-turns`/`--max-tool-calls` as defined
   in §4.1 — note today only a per-turn cap exists, `agent.py:420`),
   cancellation tests, `lqh status --json`.

## 9. Risks / open items

- The handler error-sweep (phase 1) touches many return sites in
  `handlers.py`; keep it mechanical (helper + prose preserved) and gate
  with the existing unit suite.
- Supervisor extraction (phase 4) is the riskiest refactor — the TUI must
  be moved onto the extracted component, not forked from it.
- `lqh docs agents` and the envelope both carry versions; harnesses detect
  contract changes via `schema_version` (bump on breaking change only).
- Secrets in envelopes can land in harness transcripts — documented,
  intentional (§5.4).
- Two feedback items deliberately **rejected** for scope: full `ToolSpec`
  registry (metadata + parity test instead) and deterministic
  permission-retry on resume (contextual resume instead, §4.4).
