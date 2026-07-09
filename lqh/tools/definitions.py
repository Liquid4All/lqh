from __future__ import annotations


def _tool(name: str, description: str, parameters: dict | None = None) -> dict:
    """Helper to build a single OpenAI function-calling tool definition."""
    func: dict = {
        "name": name,
        "description": description,
    }
    if parameters is not None:
        func["parameters"] = parameters
    else:
        func["parameters"] = {"type": "object", "properties": {}, "required": []}
    return {"type": "function", "function": func}


def get_all_tools(*, auto_mode: bool = False) -> list[dict]:
    """Return the list of all built-in tool definitions in OpenAI function-calling format.

    When ``auto_mode`` is True the auto-mode-only tools (``exit_auto_mode``,
    ``set_auto_stage``) are appended. They are otherwise hidden so the
    interactive agent cannot accidentally call them.
    """
    base = [
        # ------------------------------------------------------------------
        # summary
        # ------------------------------------------------------------------
        _tool(
            name="summary",
            description=(
                "Give a summary of the current project. Lists all specs, data generation "
                "scripts, evals, datasets, and training runs with timestamps and sizes. "
                "Also lists recent conversations with their first message. Use this at the "
                "start of a session to understand the project state."
            ),
        ),
        # ------------------------------------------------------------------
        # list_files
        # ------------------------------------------------------------------
        _tool(
            name="list_files",
            description=(
                "List files and directories within the project. Returns names, types "
                "(file/dir), sizes, and last-modified timestamps."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path within the project to list. "
                            "Defaults to the project root."
                        ),
                        "default": ".",
                    },
                },
                "required": [],
            },
        ),
        # ------------------------------------------------------------------
        # read_file
        # ------------------------------------------------------------------
        _tool(
            name="read_file",
            description=(
                "Read the contents of a file within the project. Supports .txt, .md, "
                ".json, .py, .jsonl as text and .parquet rendered as a table. Large files "
                "are automatically truncated; use offset and limit to page through them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the project.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Line number to start reading from (0-indexed). "
                            "Defaults to 0."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of lines to read. Defaults to all lines, "
                            "subject to the ~40 000 character truncation limit."
                        ),
                    },
                },
                "required": ["path"],
            },
        ),
        # ------------------------------------------------------------------
        # create_file
        # ------------------------------------------------------------------
        _tool(
            name="create_file",
            description=(
                "Create a new file within the project. Parent directories are created "
                "automatically. Fails if the file already exists (use write_file to overwrite). "
                "Naming: for other_specs/ use descriptive topic names (e.g. multilingual_handling.md). "
                "For data_gen/ use {task}_{version}.py (e.g. summarization_v1.py)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path for the new file within the project.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        # ------------------------------------------------------------------
        # write_file
        # ------------------------------------------------------------------
        _tool(
            name="write_file",
            description=(
                "Write or overwrite a file within the project. Creates the file and any "
                "parent directories if they do not exist; replaces contents if the file exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the project.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        # ------------------------------------------------------------------
        # edit_file
        # ------------------------------------------------------------------
        _tool(
            name="edit_file",
            description=(
                "Perform a string-replacement edit on a file within the project. The "
                "old_string must be unique in the file unless replace_all is true."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the project.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find in the file.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace old_string with.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "If true, replace every occurrence of old_string. "
                            "Defaults to false (single unique match required)."
                        ),
                        "default": False,
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        # ------------------------------------------------------------------
        # run_data_gen_pipeline
        # ------------------------------------------------------------------
        _tool(
            name="run_data_gen_pipeline",
            description=(
                "Execute a data generation pipeline script from data_gen/. The script "
                "must contain a single Pipeline subclass. The engine instantiates it "
                "per sample, calls generate(client), and writes results as parquet to "
                "datasets/<output_dataset>/data.parquet. User permission is requested "
                "before execution. "
                "Naming: use '{task}_v{N}_draft' for draft runs (~10 samples for inspection) "
                "and '{task}_v{N}' for final production runs. "
                "Example: output_dataset='summarization_v1_draft' for a test run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": (
                            "Relative path to the pipeline .py file in data_gen/."
                        ),
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": (
                            "Number of samples to generate. Use 1-3 for testing, "
                            "then scale up for production runs."
                        ),
                    },
                    "output_dataset": {
                        "type": "string",
                        "description": (
                            "Name for the output dataset directory under datasets/. "
                            "Use '{task}_v{N}_draft' for draft/inspection runs and "
                            "'{task}_v{N}' for final runs. "
                            "Example: 'summarization_v1_draft', 'summarization_v1'."
                        ),
                    },
                    "validation_instructions": {
                        "type": "string",
                        "description": (
                            "Optional path to a text file containing LLM validation "
                            "instructions for the generated samples."
                        ),
                    },
                    "samples_per_item": {
                        "type": "integer",
                        "description": (
                            "Bring-your-data mode only: how many times generate() "
                            "is called per source item. Default 1 (map once). Use "
                            "higher values when the source is small and you need "
                            "to iterate to hit num_samples (e.g. source has 200 "
                            "images, you want 2000 samples -> samples_per_item=10). "
                            "Ignored when the pipeline has no source()."
                        ),
                        "default": 1,
                    },
                },
                "required": ["script_path", "num_samples", "output_dataset"],
            },
        ),
        # ------------------------------------------------------------------
        # list_user_data
        # ------------------------------------------------------------------
        _tool(
            name="list_user_data",
            description=(
                "Report user-brought data in the project directory. Scans for: "
                "seed_data/ (JSONL/CSV/TXT seed files), images/ or other folders "
                "with image files, top-level JSONL/CSV/Parquet files (prompts or "
                "datasets). Returns filenames, row counts, and detected schemas "
                "so you can pick the right BYO-data mode without interviewing the "
                "user. Call this early in spec capture and before writing a "
                "data-gen pipeline when the user hints at bringing their own data."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # ------------------------------------------------------------------
        # run_data_filter
        # ------------------------------------------------------------------
        _tool(
            name="run_data_filter",
            description=(
                "Score a user-brought dataset and emit a filtered subset for "
                "training. Input parquet must follow the ChatML schema (messages "
                "column). Each sample is judged against the scorer file; samples "
                "scoring below threshold are dropped. Writes data.parquet (kept "
                "rows), scores.parquet (per-sample verdict), and summary.json "
                "under datasets/<output_dataset>/. Use for the bring-your-data-"
                "for-scoring workflow where the user brings 10k samples and wants "
                "to keep only the good ones."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": (
                            "Relative path to the input parquet file (user-brought)."
                        ),
                    },
                    "scorer_path": {
                        "type": "string",
                        "description": (
                            "Relative path to the scorer .md file with judging criteria."
                        ),
                    },
                    "output_dataset": {
                        "type": "string",
                        "description": (
                            "Name for the filtered output dataset under datasets/."
                        ),
                    },
                    "threshold": {
                        "type": "number",
                        "description": (
                            "Minimum score (1-10) to keep a sample. Default 6.0."
                        ),
                        "default": 6.0,
                    },
                    "model_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": (
                            "Judge model size. 'small' for fast iteration, "
                            "'medium' for production, 'large' for final filter."
                        ),
                        "default": "small",
                    },
                },
                "required": ["input_path", "scorer_path", "output_dataset"],
            },
        ),
        # ------------------------------------------------------------------
        # ask_user
        # ------------------------------------------------------------------
        _tool(
            name="ask_user",
            description=(
                "Present a question to the user in the TUI and wait for their response. "
                "If options are provided, they are shown as a selectable list (single-select "
                "by default, or multi-select with checkboxes when multi_select=true). "
                "Do NOT add an 'Other', 'Other (please specify)', 'Something else', or any "
                "free-text catch-all option yourself - the TUI ALWAYS appends its own 'Other' "
                "row automatically, and a duplicate will be shown to the user. List only the "
                "concrete choices. "
                "If options are omitted, a free-text input prompt is shown. Use this "
                "for clarification or structured questions. "
                "Use multi_select=true when the user can pick more than one option "
                "(e.g. supported languages, features to include, specs to cover)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question text to display to the user.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of selectable answer options. If omitted, "
                            "the user gets a free-text input instead."
                        ),
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": (
                            "If true, the user can select multiple options (checkboxes). "
                            "Space toggles, Enter confirms. The result is returned as a "
                            "comma-separated string. Defaults to false (single-select)."
                        ),
                        "default": False,
                    },
                },
                "required": ["question"],
            },
        ),
        # ------------------------------------------------------------------
        # show_file
        # ------------------------------------------------------------------
        _tool(
            name="show_file",
            description=(
                "Display a file's contents to the user in a formatted, scrollable TUI "
                "view. The user sees the full file, but only a truncated summary is "
                "returned to the agent's context. For .parquet dataset files, opens an "
                "interactive dataset viewer where the user can browse individual samples "
                "with keyboard navigation. Combine with ask_user to get feedback on "
                "generated data (e.g. show_file + ask_user in the same response)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the project.",
                    },
                },
                "required": ["path"],
            },
        ),
        # ------------------------------------------------------------------
        # run_scoring
        # ------------------------------------------------------------------
        _tool(
            name="run_scoring",
            description=(
                "Score a dataset using LLM-as-judge against spec-derived criteria. "
                "Two modes: (1) 'data_quality' scores existing labelled samples and "
                "writes scores.parquet alongside the dataset; (2) 'model_eval' optionally "
                "runs model inference on unlabelled prompts, then scores the outputs, "
                "writing results to evals/runs/<run_name>/. "
                "Requires a scorer .md file in evals/scorers/. Create the scorer first "
                "using create_file with criteria derived from the spec(s)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": (
                            "Relative path to the dataset directory (e.g. "
                            "'datasets/summarization_v1_draft'). Must contain data.parquet."
                        ),
                    },
                    "scorer": {
                        "type": "string",
                        "description": (
                            "Relative path to the scorer .md file "
                            "(e.g. 'evals/scorers/summarization_v1.md')."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["data_quality", "model_eval"],
                        "description": (
                            "'data_quality' scores the dataset's existing assistant turns "
                            "and writes scores.parquet next to data.parquet. "
                            "'model_eval' strips assistant turns, runs inference, scores "
                            "the model outputs, and writes to evals/runs/<run_name>/. "
                            "For model_eval of a base/zero-shot model, ALWAYS supply a "
                            "system prompt (system_prompt_path or inference_system_prompt); "
                            "without one a base model is confused and scores near zero."
                        ),
                    },
                    "run_name": {
                        "type": "string",
                        "description": (
                            "Name for the eval run directory under evals/runs/. "
                            "Required for mode='model_eval'. Use descriptive names like "
                            "'baseline_zero_shot', 'post_training_run_001'."
                        ),
                    },
                    "model_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": (
                            "Size of the scoring LLM judge. 'small' (default) is fast "
                            "and sufficient for ~80%% of tasks. 'medium' for harder "
                            "tasks (~95%% coverage). 'large' for very nuanced scoring."
                        ),
                        "default": "small",
                    },
                    "inference_model": {
                        "type": "string",
                        "description": (
                            "Baseline model to run inference with in mode='model_eval'. "
                            "Use a pool/baseline name: 'small', 'medium', 'large', "
                            "'orchestration', or 'random:small:seed123'. "
                            "NOTE: Liquid checkpoints (e.g. LiquidAI/LFM2.5-1.2B-Instruct) "
                            "CANNOT be evaluated here — the router.liquid.ai API is retired; "
                            "use eval_hf_model or start_local_eval for those. "
                            "Required for mode='model_eval'."
                        ),
                    },
                    "inference_system_prompt": {
                        "type": "string",
                        "description": (
                            "System prompt to prepend to each sample before running "
                            "inference. Used in mode='model_eval' to test different "
                            "prompt strategies with the same model and eval dataset. "
                            "Always set this (or system_prompt_path) for a base-model "
                            "baseline — a well-structured prompt with task instructions "
                            "and expected output format; without one the baseline is "
                            "a confused, near-zero score."
                        ),
                    },
                    "system_prompt_path": {
                        "type": "string",
                        "description": (
                            "Relative path to a system prompt .md file "
                            "(e.g. 'prompts/summarization_v1.md'). If provided "
                            "without inference_system_prompt, the file's content "
                            "is used as the system prompt. Stored in config.json "
                            "for traceability."
                        ),
                    },
                    "response_format_path": {
                        "type": "string",
                        "description": (
                            "Relative path to a JSON schema file for structured "
                            "output (e.g. 'prompts/translation.schema.json'). "
                            "If omitted and system_prompt_path is set, auto-discovers "
                            "{task}.schema.json in the same directory."
                        ),
                    },
                },
                "required": ["dataset", "scorer", "mode"],
            },
        ),
        # ------------------------------------------------------------------
        # list_models
        # ------------------------------------------------------------------
        _tool(
            name="list_models",
            description=(
                "List the Liquid AI model catalog (HuggingFace IDs + kind: "
                "base/instruct/thinking) plus the baseline/judge pool models. "
                "Liquid checkpoints are evaluated via the HuggingFace inference "
                "path (eval_hf_model / start_local_eval), not via the API. The "
                "pool models (small/medium/large/orchestration) are the only "
                "names usable as inference_model in run_scoring mode='model_eval'."
            ),
        ),
        # ------------------------------------------------------------------
        # get_eval_failures
        # ------------------------------------------------------------------
        _tool(
            name="get_eval_failures",
            description=(
                "Extract failure cases from an eval run's results. Returns the "
                "lowest-scoring samples with their messages, scores, and judge "
                "reasoning. Uses a hybrid approach: all samples below a score "
                "threshold, padded with bottom-N lowest scorers to ensure a "
                "minimum number of failures. Use after run_scoring to identify "
                "what went wrong and guide prompt refinement."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "eval_run": {
                        "type": "string",
                        "description": (
                            "Relative path to the eval run directory "
                            "(e.g. 'evals/runs/prompt_v1_iter1'). "
                            "Must contain results.parquet."
                        ),
                    },
                    "threshold": {
                        "type": "number",
                        "description": (
                            "Score threshold: samples scoring strictly below this "
                            "are considered failures. Default: 6.0."
                        ),
                        "default": 6.0,
                    },
                    "min_failures": {
                        "type": "integer",
                        "description": (
                            "Minimum number of failure samples to return. If fewer "
                            "than this score below threshold, the lowest-scoring "
                            "samples are added. Default: 5."
                        ),
                        "default": 5,
                    },
                    "max_failures": {
                        "type": "integer",
                        "description": (
                            "Maximum number of failure samples to return. Default: 15."
                        ),
                        "default": 15,
                    },
                },
                "required": ["eval_run"],
            },
        ),
        # ------------------------------------------------------------------
        # list_skills
        # ------------------------------------------------------------------
        _tool(
            name="list_skills",
            description=(
                "List all available skills (modes) with their names and descriptions. "
                "Use this to discover what skills are available before loading one."
            ),
        ),
        # ------------------------------------------------------------------
        # load_skill
        # ------------------------------------------------------------------
        _tool(
            name="load_skill",
            description=(
                "Load a skill's SKILL.md instructions into the current conversation as "
                "a system message. This changes the agent's behavior according to the "
                "skill's guidelines. Use list_skills first to see available options."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": (
                            "Name of the skill to load (e.g., 'spec_capture', "
                            "'data_generation', 'data_validation')."
                        ),
                    },
                },
                "required": ["skill_name"],
            },
        ),
        # ------------------------------------------------------------------
        # hf_push
        # ------------------------------------------------------------------
        _tool(
            name="hf_push",
            description=(
                "Push a local directory to Hugging Face Hub as either a dataset "
                "(folder containing .parquet files) or a model (folder containing "
                "config.json + *.safetensors/*.bin/*.ckpt/*.pt). The repo type is "
                "auto-detected from the folder contents; pass repo_type to override. "
                "If the folder contains a README.md, it is uploaded as the repo card "
                "— this is the recommended way to attach documentation. Creates the "
                "repo if it does not exist (private by default). Requires HF_TOKEN "
                "env var. User permission is requested before pushing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "local_path": {
                        "type": "string",
                        "description": (
                            "Relative path to the local directory to push "
                            "(e.g. 'datasets/summarization_v1' or "
                            "'runs/sft-1/checkpoints/step_500'). Must be a folder "
                            "containing either parquet files (dataset) or HF-style "
                            "model files (config.json + weights)."
                        ),
                    },
                    "repo_type": {
                        "type": "string",
                        "enum": ["dataset", "model"],
                        "description": (
                            "Override the auto-detected repo type. By default the "
                            "type is inferred from the folder contents (parquet → "
                            "dataset, config.json/weights → model)."
                        ),
                    },
                    "repo_id": {
                        "type": "string",
                        "description": (
                            "HF repo ID (e.g. 'username/my-dataset'). If omitted, "
                            "auto-generated from HF username + project dir + folder "
                            "name. Use hf_repo_info first to discover your username."
                        ),
                    },
                    "private": {
                        "type": "boolean",
                        "description": (
                            "Whether the repo should be private. Defaults to true."
                        ),
                        "default": True,
                    },
                    "split": {
                        "type": "string",
                        "description": (
                            "Dataset split name (e.g. 'train', 'test', 'validation'). "
                            "Defaults to 'train'. Only applies when repo_type=dataset."
                        ),
                        "default": "train",
                    },
                    "subset": {
                        "type": "string",
                        "description": (
                            "Dataset config/subset name. Use this to push multiple "
                            "related datasets as subsets of a single HF repo. Only "
                            "applies when repo_type=dataset."
                        ),
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message for the push.",
                    },
                },
                "required": ["local_path"],
            },
        ),
        # ------------------------------------------------------------------
        # hf_pull
        # ------------------------------------------------------------------
        _tool(
            name="hf_pull",
            description=(
                "Download a dataset or model from Hugging Face Hub to local storage. "
                "The repo type is auto-detected by querying the Hub (model first, "
                "then dataset); pass repo_type to override. Datasets are saved as "
                "parquet under datasets/{repo-name}/; models are saved with their "
                "full file tree under models/{repo-name}/ — the local path can then "
                "be used as base_model in training configs or as the eval target. "
                "Passes HF_TOKEN if available for private repos."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": (
                            "HF repo ID (e.g. 'meta-llama/Llama-3.2-1B' or "
                            "'username/my-dataset')."
                        ),
                    },
                    "repo_type": {
                        "type": "string",
                        "enum": ["dataset", "model"],
                        "description": (
                            "Override the auto-detected repo type. By default the "
                            "type is inferred by querying the Hub."
                        ),
                    },
                    "local_path": {
                        "type": "string",
                        "description": (
                            "Where to save locally, relative to project dir. "
                            "Defaults to 'datasets/{repo-name}/' for datasets and "
                            "'models/{repo-name}/' for models."
                        ),
                    },
                    "split": {
                        "type": "string",
                        "description": (
                            "Specific split to download (e.g. 'train'). "
                            "If omitted, downloads all splits. Only applies when "
                            "repo_type=dataset."
                        ),
                    },
                    "subset": {
                        "type": "string",
                        "description": (
                            "Dataset config/subset name. Required if the dataset "
                            "has multiple configs. Only applies when repo_type=dataset."
                        ),
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Specific files to download instead of the full repo "
                            "(e.g. ['data/train-00000-of-00001.parquet'] or "
                            "['config.json', 'tokenizer.json']). Useful for "
                            "lightweight inspection without pulling weights."
                        ),
                    },
                },
                "required": ["repo_id"],
            },
        ),
        # ------------------------------------------------------------------
        # hf_repo_info
        # ------------------------------------------------------------------
        _tool(
            name="hf_repo_info",
            description=(
                "Get info about a HF repo or the authenticated user. "
                "Call with no arguments to get the current user's username, "
                "orgs, and token scope — useful before constructing repo IDs "
                "for hf_push. Call with repo_id to inspect a specific repo."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": (
                            "HF repo ID to inspect. If omitted, returns info "
                            "about the authenticated user (whoami)."
                        ),
                    },
                    "repo_type": {
                        "type": "string",
                        "enum": ["dataset", "model"],
                        "description": (
                            "Type of repo to inspect. Defaults to 'dataset'."
                        ),
                        "default": "dataset",
                    },
                },
                "required": [],
            },
        ),
        # ------------------------------------------------------------------
        # pull
        # ------------------------------------------------------------------
        _tool(
            name="pull",
            description=(
                "Download a model, dataset, or artifact into local storage using a "
                "location URI. The scheme is explicit and never guessed:\n"
                "  - 'hf:owner/repo[@rev]' — Hugging Face Hub (models under models/, "
                "datasets under datasets/).\n"
                "  - 'lqh:<artifact_id>' — an LQH cloud artifact in R2 (checkpoints "
                "arrive as a .tar.gz).\n"
                "For private HF repos, set HF_TOKEN. Use 'artifacts' (action=list) to "
                "find artifact IDs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Location URI to pull from: 'hf:owner/repo[@rev]' or "
                            "'lqh:<artifact_id>'."
                        ),
                    },
                    "dest": {
                        "type": "string",
                        "description": (
                            "Local path (relative to the project) to save into. "
                            "Defaults to models/ or datasets/ for hf:, and "
                            "artifacts/<id> for lqh:."
                        ),
                    },
                },
                "required": ["source"],
            },
        ),
        # ------------------------------------------------------------------
        # push
        # ------------------------------------------------------------------
        _tool(
            name="push",
            description=(
                "Upload to Hugging Face Hub from a location URI. The destination must "
                "be an 'hf:owner/repo'. The source is either:\n"
                "  - a local path — uploaded directly (dataset or model, auto-detected); "
                "or\n"
                "  - 'lqh:<artifact_id>' — an R2 artifact, transferred to HF by a short "
                "CPU-only cloud sandbox (bytes never round-trip through this machine; "
                "requires a stored HF token via /hf_login).\n"
                "Creates the repo if missing (private by default)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "What to push: a local path (e.g. 'runs/sft-1/model') or "
                            "'lqh:<artifact_id>'."
                        ),
                    },
                    "dest": {
                        "type": "string",
                        "description": "Destination repo as 'hf:owner/repo'.",
                    },
                    "private": {
                        "type": "boolean",
                        "description": "Whether the HF repo should be private. Defaults to true.",
                        "default": True,
                    },
                },
                "required": ["source", "dest"],
            },
        ),
        # ------------------------------------------------------------------
        # gguf_convert
        # ------------------------------------------------------------------
        _tool(
            name="gguf_convert",
            description=(
                "Convert an LQH checkpoint artifact to GGUF (the llama.cpp / "
                "local-inference format) and quantize it, via a short CPU-only "
                "cloud sandbox. Give it a checkpoint artifact id and one or more "
                "quant types; it converts once to f16 and quantizes into each "
                "type, registering every .gguf back as a downloadable artifact "
                "(kind 'gguf'). If the checkpoint is a LoRA adapter it is merged "
                "onto its base first (auto-detected from lineage; pass base_model "
                "if it isn't recorded). Optionally also pushes the files to a HF "
                "repo (requires a stored HF token via /hf_login). Use 'artifacts' "
                "(action=list) to find checkpoint ids; check training_status for "
                "progress. Common quant picks: Q4_K (fast, good), Q8_0 (near-"
                "lossless, ~2x slower)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": (
                            "The LQH checkpoint artifact id to convert "
                            "(a UUID; from 'artifacts' list)."
                        ),
                    },
                    "quant_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0"],
                        },
                        "description": (
                            "One or more quantization types to produce. "
                            "Q4_0: fastest, lower quality (poor on <1B models); "
                            "Q4_K: fast, better quality; Q5_K/Q6_K: better quality, "
                            "slower and often lacking fast kernels; Q8_0: near "
                            "full-precision, ~2x slower than Q4."
                        ),
                    },
                    "target_hf_repo": {
                        "type": "string",
                        "description": (
                            "Optional 'owner/repo' to also push the produced .gguf "
                            "files to on Hugging Face. Requires a stored HF token."
                        ),
                    },
                    "private": {
                        "type": "boolean",
                        "description": (
                            "Whether the HF repo should be private (only used with "
                            "target_hf_repo). Defaults to true."
                        ),
                        "default": True,
                    },
                    "include_f16": {
                        "type": "boolean",
                        "description": (
                            "Also register the intermediate f16 GGUF (unquantized). "
                            "Defaults to false."
                        ),
                        "default": False,
                    },
                    "base_model": {
                        "type": "string",
                        "description": (
                            "HF base model id (org/repo) to merge a LoRA adapter "
                            "onto before conversion. Only needed when the adapter's "
                            "lineage doesn't already record its base."
                        ),
                    },
                    "artifact_format": {
                        "type": "string",
                        "enum": ["lora", "full"],
                        "description": (
                            "Override LoRA/full detection. Usually derived from the "
                            "checkpoint's lineage; set 'lora' to force a merge for a "
                            "pre-lineage adapter whose filename doesn't match the "
                            "heuristic (pair it with base_model)."
                        ),
                    },
                },
                "required": ["artifact_id", "quant_types"],
            },
        ),
        # ------------------------------------------------------------------
        # artifacts
        # ------------------------------------------------------------------
        _tool(
            name="artifacts",
            description=(
                "Manage the cloud artifacts (checkpoints, predictions, metrics, logs) "
                "registered for this project. Actions:\n"
                "  - list (default): show artifacts with size, expiry, pin status.\n"
                "  - pin: keep an artifact indefinitely (exempt from auto-expiry).\n"
                "  - unpin: re-arm the per-kind expiry clock.\n"
                "  - delete: remove an artifact (R2 bytes purged on the next retention "
                "tick).\n"
                "Unpinned artifacts auto-expire per a retention policy; the best/final "
                "checkpoints and referenced artifacts are protected automatically — pin "
                "anything else you want to keep."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "pin", "unpin", "delete"],
                        "description": "What to do. Defaults to 'list'.",
                        "default": "list",
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": "Artifact ID — required for pin/unpin/delete.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": [
                            "checkpoint", "predictions", "metrics", "logs",
                            "eval_result", "dataset", "bundle", "other",
                        ],
                        "description": "Filter the list by kind (list action only).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max artifacts to list. Defaults to 50.",
                        "default": 50,
                    },
                },
                "required": [],
            },
        ),
        # ------------------------------------------------------------------
        # push_to_production
        # ------------------------------------------------------------------
        _tool(
            name="push_to_production",
            description=(
                "Deploy a trained checkpoint artifact as a live inference endpoint "
                "on LQH Cloud. The model is served OpenAI-compatible at "
                "https://inference.lqh.ai/v1 with the deployment name as the model id. "
                "LoRA checkpoint artifacts are auto-merged into their base model "
                "first (status: pending → merging → deploying → running); full "
                "fine-tunes skip the merge (pending → deploying → running). Billing "
                "is per GPU-hour while the deployment runs. Use this after training "
                "when the user wants to serve the model — find the checkpoint's "
                "artifact ID via 'artifacts' (action=list), then create an access "
                "key with create_inference_key so the user can call the endpoint."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": (
                            "ID of the checkpoint artifact to deploy (from "
                            "'artifacts' action=list). LoRA and full checkpoints "
                            "both work."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Deployment name — this becomes the model id clients "
                            "pass to the OpenAI-compatible endpoint. Must be unique; "
                            "use a short descriptive slug (e.g. 'summarizer-v2')."
                        ),
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["debug", "prod"],
                        "description": (
                            "'debug' (default) for cheap testing; 'prod' for "
                            "customer-facing traffic. Only use 'prod' when the user "
                            "explicitly asks for a production deployment."
                        ),
                        "default": "debug",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": (
                            "GPU type to serve on. Omit to let the backend pick a "
                            "sensible default for the model size."
                        ),
                    },
                    "min_containers": {
                        "type": "integer",
                        "description": (
                            "Minimum number of serving containers. 0 allows "
                            "scale-to-zero (cold starts); omit for the default."
                        ),
                    },
                    "max_containers": {
                        "type": "integer",
                        "description": (
                            "Maximum number of serving containers for autoscaling. "
                            "Omit for the default."
                        ),
                    },
                    "project_id": {
                        "type": "string",
                        "description": (
                            "Project to attribute the deployment to. Defaults to "
                            "the current project."
                        ),
                    },
                    "artifact_format": {
                        "type": "string",
                        "enum": ["lora", "full"],
                        "description": (
                            "Override the artifact classification when its lineage "
                            "is missing or wrong (e.g. a LoRA adapter that was "
                            "published as model.tar.gz). Usually omit."
                        ),
                    },
                    "base_model": {
                        "type": "string",
                        "description": (
                            "HF id the adapter merges onto at boot, e.g. "
                            "'LiquidAI/LFM2.5-1.2B-Instruct'. Required for LoRA "
                            "artifacts whose lineage lacks base_model; ignored for "
                            "full checkpoints."
                        ),
                    },
                },
                "required": ["artifact_id", "name"],
            },
        ),
        # ------------------------------------------------------------------
        # list_deployments
        # ------------------------------------------------------------------
        _tool(
            name="list_deployments",
            description=(
                "List all inference deployments for the account: name, status, "
                "tier, GPU, estimated $/hr, and billed cost to date. Use this to "
                "check what is currently serving (and costing money) before "
                "creating, stopping, or restarting deployments."
            ),
        ),
        # ------------------------------------------------------------------
        # get_deployment
        # ------------------------------------------------------------------
        _tool(
            name="get_deployment",
            description=(
                "Get one deployment's full status plus a current-period usage "
                "summary (requests, errors, tokens, latency, GPU cost). Use this "
                "to track a deployment through its merge/deploy phases after "
                "push_to_production, or to report traffic and spend to the user."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "deployment_id": {
                        "type": "string",
                        "description": "ID of the deployment to inspect.",
                    },
                },
                "required": ["deployment_id"],
            },
        ),
        # ------------------------------------------------------------------
        # stop_deployment
        # ------------------------------------------------------------------
        _tool(
            name="stop_deployment",
            description=(
                "Stop a running deployment. GPU billing stops; the deployment's "
                "name and configuration are kept so it can be brought back with "
                "restart_deployment. Use when the user is done testing or wants "
                "to cut costs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "deployment_id": {
                        "type": "string",
                        "description": "ID of the deployment to stop.",
                    },
                },
                "required": ["deployment_id"],
            },
        ),
        # ------------------------------------------------------------------
        # restart_deployment
        # ------------------------------------------------------------------
        _tool(
            name="restart_deployment",
            description=(
                "Restart a stopped (or errored) deployment. GPU billing resumes "
                "once it is running again. The endpoint and model name stay the "
                "same, so existing inference keys keep working."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "deployment_id": {
                        "type": "string",
                        "description": "ID of the deployment to restart.",
                    },
                },
                "required": ["deployment_id"],
            },
        ),
        # ------------------------------------------------------------------
        # create_inference_key
        # ------------------------------------------------------------------
        _tool(
            name="create_inference_key",
            description=(
                "Create an inference API key (lqh_inf_...) for the customer-facing "
                "endpoint https://inference.lqh.ai/v1. The plaintext key is returned "
                "ONCE in this call and can never be retrieved again — relay it to "
                "the user immediately. By default the key grants access to all of "
                "the org's deployments; pass deployment_ids to scope it to "
                "specific ones. Use after push_to_production so the user can "
                "actually call their deployed model."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Human-readable name for the key (e.g. 'staging', "
                            "'acme-customer')."
                        ),
                    },
                    "deployment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Deployment IDs this key may access. If omitted, the "
                            "key is created with access to all deployments."
                        ),
                    },
                    "all_deployments": {
                        "type": "boolean",
                        "description": (
                            "Grant access to every deployment (including future "
                            "ones). Defaults to true when deployment_ids is omitted."
                        ),
                    },
                },
                "required": ["name"],
            },
        ),
        # ------------------------------------------------------------------
        # list_inference_keys
        # ------------------------------------------------------------------
        _tool(
            name="list_inference_keys",
            description=(
                "List the org's inference API keys: name, prefix, scope, and "
                "revocation status. Plaintext keys are never shown here — only "
                "create_inference_key returns one, and only once."
            ),
        ),
        # ------------------------------------------------------------------
        # revoke_inference_key
        # ------------------------------------------------------------------
        _tool(
            name="revoke_inference_key",
            description=(
                "Revoke an inference API key immediately. Requests using it will "
                "start failing with 401. This cannot be undone — create a new key "
                "if access is needed again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key_id": {
                        "type": "string",
                        "description": "ID of the inference key to revoke.",
                    },
                },
                "required": ["key_id"],
            },
        ),
        # ------------------------------------------------------------------
        # start_training
        # ------------------------------------------------------------------
        _tool(
            name="start_training",
            description=(
                "Start a fine-tuning run as a background subprocess. Supports SFT "
                "(supervised fine-tuning) and on-policy DPO (direct preference "
                "optimization). Training runs in a separate process with GPU/torch, "
                "while the agent stays responsive. Progress is tracked via the "
                "filesystem. Requires the 'train' optional dependencies "
                "(pip install lqh[train]). User permission is requested before starting. "
                "The compute target (LQH Cloud vs a bring-your-own-compute remote) is "
                "fixed per project and chosen once via a system picker — do NOT ask the "
                "user where to train and do NOT pass any compute/remote argument; just "
                "call start_training and it routes automatically. "
                "By default a small hyperparameter sweep runs (`enable_sweep=true`) — "
                "the right choice for most users: on a new dataset you don't know how "
                "this model size × dataset combination behaves, so several lr/epoch "
                "configs train and the winner is picked on the held-out signal. Set "
                "`enable_sweep=false` for a single run when the working hyperparameters "
                "are already known from a previous sweep, when the user explicitly "
                "specifies them, or for quick test/smoke runs — then set "
                "`learning_rate` and `num_epochs` manually. "
                "`eval_dataset` is REQUIRED either way (sweep winner selection and "
                "judge scoring both need it). "
                "You must also pass `scorer` — set to the project's default/best scorer — "
                "so the best checkpoint gets a real judge score; the call is rejected "
                "unless you pass `scorer` or set `disable_scoring=true` (only when the "
                "user explicitly asks not to score). The judge score is not returned by "
                "this call — after the run completes, fetch it with `training_status`. "
                "Vision-language bases (LFM2.5-VL-*) are supported for SFT only (not "
                "DPO): the run automatically switches to the VLM path (image-aware "
                "collation, the VLM LoRA recipe). The dataset must carry its images "
                "inline as image_url data-URLs in the messages (the standard vision "
                "data-gen output). Token budget note: each image costs up to "
                "training.max_image_tokens (default 256) of the max_seq_length budget."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["sft", "on_policy_dpo"],
                        "description": (
                            "'sft' for supervised fine-tuning, 'on_policy_dpo' for "
                            "on-policy direct preference optimization."
                        ),
                    },
                    "base_model": {
                        "type": "string",
                        "description": (
                            "HuggingFace model ID (e.g. 'LiquidAI/LFM2.5-1.2B-Instruct') "
                            "or local path to a model directory (e.g. 'runs/run_001/model')."
                        ),
                    },
                    "dataset": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "path": {"type": "string"},
                                                "repeat": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                },
                                            },
                                            "required": ["path"],
                                            "additionalProperties": False,
                                        },
                                    ]
                                },
                            },
                        ],
                        "description": (
                            "Training data source(s). EITHER a single relative path to a "
                            "dataset directory containing data.parquet "
                            "(e.g. 'datasets/summarization_v1'), OR a LIST of such paths to "
                            "combine into one training set "
                            "(e.g. ['datasets/type_a', 'datasets/type_b']). Every source "
                            "must be a filtered/scored dataset you have inspected — same "
                            "rule as a single-source run; never list a raw unfiltered set. "
                            "Use a list to (1) train on ALL the good data you have rather "
                            "than just one file — when you've accumulated batches over time "
                            "(scale-up: a 2k file, then a 10k file) train on every batch, "
                            "don't discard the smaller earlier ones; (2) mix your own data "
                            "with public/private HuggingFace datasets (run `pull`/`hf_pull` "
                            "first, filter it, then reference the local 'datasets/<repo>' "
                            "dir); (3) blend different data TYPES from separate pipelines "
                            "(type-A + type-B). "
                            "Default is plain 1:1 concatenation — this is what you want for "
                            "case (1): same kind of data in batches, give every source the "
                            "default repeat of 1. Only reach for `repeat` to fix an "
                            "imbalance between different TYPES (case 3): pass an object "
                            "{'path': 'datasets/type_a', 'repeat': 5} to over-sample the "
                            "smaller type so the per-batch mix is balanced (e.g. 2k type-A "
                            "vs 10k type-B → repeat type-A ~5×). `repeat` is integer "
                            "over-sampling of the same rows and is orthogonal to num_epochs "
                            "(which controls passes over the whole blend). This is the "
                            "source of training prompts only — SFT trains on these "
                            "conversations, DPO generates on-policy rollouts from them; "
                            "never used for evaluation — that is `eval_dataset`."
                        ),
                    },
                    "eval_dataset": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": (
                            "Held-out eval source(s). EITHER a single relative path to an "
                            "eval dataset directory (must contain data.parquet), OR a LIST "
                            "of such paths. REQUIRED, and each source must be DISTINCT from "
                            "every training source. Unlike `dataset`, eval sources are kept "
                            "SEPARATE: the best checkpoint is judge-scored on each source "
                            "independently, producing a per-source score plus a combined "
                            "MACRO-AVERAGE headline (each source weighted equally, "
                            "regardless of size) — pass a list "
                            "(e.g. ['datasets/type_a_eval', 'datasets/type_b_eval']) when "
                            "you want to measure each sub-task separately rather than only "
                            "an aggregate. For sweep winner selection the in-training "
                            "val_loss is computed over the CONCATENATION of all eval "
                            "sources (a single scalar); for DPO the sweep proxy is a "
                            "preference split instead. (No `repeat` here — weighting is "
                            "meaningless for eval.)"
                        ),
                    },
                    "scorer": {
                        "type": "string",
                        "description": (
                            "Relative path to the scorer .md file. Set this to the "
                            "project's default or currently-best scorer — typically the "
                            "one under evals/scorers/ used for the baseline eval "
                            "(e.g. 'evals/scorers/summarization_v1.md'). REQUIRED unless "
                            "you set `disable_scoring=true`: the call is rejected if "
                            "neither is provided. Without a scorer the run yields only the "
                            "val_loss proxy and NO judge score on the best checkpoint. The "
                            "judge score is NOT returned by this call — after the run "
                            "finishes, fetch it via `training_status` and the run's "
                            "eval_result.json / sweep_summary.json."
                        ),
                    },
                    "disable_scoring": {
                        "type": "boolean",
                        "description": (
                            "SFT only. Set true ONLY when the user explicitly asks not to "
                            "score the run ('just train, no eval', 'skip scoring'). When "
                            "false (default) you MUST pass `scorer` — the call is rejected "
                            "if neither `scorer` nor `disable_scoring=true` is given, so "
                            "that skipping the judge score is always a deliberate choice, "
                            "never a silent omission. DPO ignores this and rejects it: "
                            "on-policy DPO must score rollouts every iteration to build "
                            "preferences, so a scorer is always required for DPO."
                        ),
                        "default": False,
                    },
                    "run_name": {
                        "type": "string",
                        "description": (
                            "Name for the run directory under runs/. Auto-generated if omitted "
                            "(e.g. 'sft_001', 'dpo_002')."
                        ),
                    },
                    "lora": {
                        "type": "boolean",
                        "description": (
                            "Whether to use LoRA for parameter-efficient fine-tuning. "
                            "Defaults to true."
                        ),
                        "default": True,
                    },
                    "enable_sweep": {
                        "type": "boolean",
                        "description": (
                            "Default true: train a small hyperparameter grid "
                            "(SFT: lr ∈ {2e-5, 5e-5, 1e-4} × epochs ∈ {2, 3}) and keep "
                            "the winner — use this whenever you don't yet know how this "
                            "model size × dataset combination performs (i.e. most new "
                            "datasets). Set false for a single training run when good "
                            "hyperparameters are already known (e.g. from an earlier "
                            "sweep on this dataset), when the user explicitly prescribes "
                            "them, or for test/smoke runs where speed matters more than "
                            "tuning. With false, set `learning_rate` and `num_epochs` "
                            "yourself — reasonable SFT starting point: a medium lr of "
                            "5e-5 and 2 epochs."
                        ),
                        "default": True,
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": (
                            "Number of training epochs (SFT only). Only takes effect "
                            "with `enable_sweep=false` (the sweep grid overrides it). "
                            "Recommended single-run value: 2. Default: 3."
                        ),
                        "default": 3,
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": (
                            "Learning rate. Only takes effect with `enable_sweep=false` "
                            "(the sweep grid overrides it). Recommended single-run "
                            "value: 5e-5 (medium) for SFT; default if omitted: 2e-5 for "
                            "SFT, 5e-6 for DPO."
                        ),
                    },
                    "num_iterations": {
                        "type": "integer",
                        "description": (
                            "Number of on-policy iterations (DPO only). Default: 5."
                        ),
                        "default": 5,
                    },
                    "dpo_beta": {
                        "type": "number",
                        "description": "DPO beta parameter (DPO only). Default: 0.1.",
                        "default": 0.1,
                    },
                    "golden_source": {
                        "type": "string",
                        "enum": ["dataset", "api"],
                        "description": (
                            "Where chosen trajectories come from for DPO. 'dataset' uses "
                            "original assistant turns (free). 'api' generates better responses "
                            "via the API. Default: 'dataset'."
                        ),
                        "default": "dataset",
                    },
                },
                "required": ["type", "base_model", "dataset", "eval_dataset"],
            },
        ),
        # ------------------------------------------------------------------
        # training_status
        # ------------------------------------------------------------------
        _tool(
            name="training_status",
            description=(
                "Check the status of training runs. Shows current step, loss, "
                "learning rate, eval scores, and whether the subprocess is alive. "
                "If run_name is omitted, shows status of all runs. Running training "
                "jobs are watched in the background, so do not repeatedly poll this "
                "tool. If you need to wait for training to finish, end the "
                "conversation without another tool call; the session will wake "
                "automatically when the watcher observes completion."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": (
                            "Name of a specific run to check. If omitted, shows all runs."
                        ),
                    },
                },
                "required": [],
            },
        ),
        # ------------------------------------------------------------------
        # stop_training
        # ------------------------------------------------------------------
        _tool(
            name="stop_training",
            description=(
                "Stop a running training subprocess. Sends SIGTERM for graceful "
                "shutdown, then SIGKILL if it doesn't exit within 10 seconds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of the training run to stop.",
                    },
                },
                "required": ["run_name"],
            },
        ),
        # ------------------------------------------------------------------
        # start_local_eval
        # ------------------------------------------------------------------
        _tool(
            name="start_local_eval",
            description=(
                "Run model inference as a subprocess and score the results "
                "via the API judge. Best for evaluating a checkpoint that "
                "lives on a local filesystem you control — the current "
                "machine or the project's configured bring-your-own-compute "
                "SSH remote. The compute target is fixed per project (chosen "
                "once via the system picker); do NOT pass any compute/remote "
                "argument. For models trained on LQH Cloud the checkpoint "
                "lives in R2 and is not yet directly evaluable here — push "
                "the model to HuggingFace via hf_push and use eval_hf_model "
                "instead. Output lives at runs/<run_name>/ — "
                "predictions.parquet plus eval_result.json once scoring "
                "finishes (NOT under evals/runs/, which is for run_scoring's "
                "API-mode evals)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": (
                            "Relative path to the model directory "
                            "(e.g. 'runs/run_001/model' or a checkpoint dir)."
                        ),
                    },
                    "dataset": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": (
                            "Eval dataset source(s): a single relative path to an "
                            "eval dataset directory (must contain data.parquet), OR a "
                            "list of such paths. Multiple sources are scored SEPARATELY "
                            "and combined into a macro-average headline (each source "
                            "weighted equally) — the per-source breakdown is in the run's "
                            "eval_result.json. Use a list to measure each sub-task "
                            "(e.g. ['datasets/type_a_eval', 'datasets/type_b_eval'])."
                        ),
                    },
                    "scorer": {
                        "type": "string",
                        "description": (
                            "Relative path to the scorer .md file "
                            "(e.g. 'evals/scorers/summarization_v1.md')."
                        ),
                    },
                    "run_name": {
                        "type": "string",
                        "description": (
                            "Name for the eval run directory under evals/runs/. "
                            "Auto-generated if omitted."
                        ),
                    },
                    "system_prompt_path": {
                        "type": "string",
                        "description": (
                            "Relative path to a .md file whose contents are "
                            "prepended as a system message before each sample. "
                            "Mirrors the run_scoring system_prompt_path argument "
                            "so local and API evals run with the same instructions."
                        ),
                    },
                    "response_format_path": {
                        "type": "string",
                        "description": (
                            "Relative path to a JSON-schema file used to "
                            "constrain decoding (requires lm-format-enforcer). "
                            "If omitted but system_prompt_path is set, "
                            "auto-discovers prompts/<task>.schema.json."
                        ),
                    },
                    "max_new_tokens": {
                        "type": "integer",
                        "description": (
                            "Token cap for each generated response. Default "
                            "4096; bump higher for long-form outputs (thread "
                            "translations, multi-paragraph summaries)."
                        ),
                        "default": 4096,
                    },
                },
                "required": ["model_path", "dataset", "scorer"],
            },
        ),
        # ------------------------------------------------------------------
        # eval_hf_model
        # ------------------------------------------------------------------
        _tool(
            name="eval_hf_model",
            description=(
                "Evaluate any HuggingFace checkpoint on a project's eval set "
                "via LQH Cloud. Use this to score a public/private HF model "
                "(LoRA adapter or full checkpoint) without training it "
                "locally — e.g. evaluating someone else's fine-tune, "
                "benchmarking an off-the-shelf base, or running an "
                "alternative checkpoint version against the same scorer. "
                "Generates rollouts on a GPU sandbox and scores them via "
                "the LLM judge in one job; result lands as eval_result.json "
                "under runs/<run_name>/."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": (
                            "HuggingFace repository id "
                            "(e.g. 'Qwen/Qwen3.5-3B-Instruct' or "
                            "'someuser/my-translation-lora')."
                        ),
                    },
                    "revision": {
                        "type": "string",
                        "description": (
                            "Git revision / branch / tag to fetch. "
                            "Defaults to 'main'."
                        ),
                        "default": "main",
                    },
                    "training_method": {
                        "type": "string",
                        "enum": ["lora", "full"],
                        "description": (
                            "'lora' for an adapter repo (requires base_model); "
                            "'full' for a merged/standalone checkpoint. Use 'full' "
                            "for an off-the-shelf Liquid catalog model (e.g. "
                            "LiquidAI/LFM2.5-1.2B-Instruct or a -Base checkpoint) — "
                            "those are full checkpoints, not adapters."
                        ),
                        "default": "lora",
                    },
                    "base_model": {
                        "type": "string",
                        "description": (
                            "HF repo id of the base model. Required when "
                            "training_method='lora' — pinned regardless of "
                            "what adapter_config.json says. Ignored for "
                            "'full'."
                        ),
                    },
                    "eval_dataset": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": (
                            "Eval dataset source(s): a single relative path to an "
                            "eval dataset directory (must contain data.parquet), OR a "
                            "list of such paths. Multiple sources are scored separately "
                            "and combined into a macro-average headline (each source "
                            "weighted equally); the per-source breakdown lands in the "
                            "run's eval_result.json."
                        ),
                    },
                    "scorer": {
                        "type": "string",
                        "description": (
                            "Relative path to the scorer .md file."
                        ),
                    },
                    "system_prompt_path": {
                        "type": "string",
                        "description": (
                            "Optional relative path to a .md file whose "
                            "contents are prepended as a system message "
                            "before each sample. Auto-discovers a sibling "
                            "<task>.schema.json for constrained decoding."
                        ),
                    },
                    "judge_size": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": (
                            "Which judge to use for scoring. Default 'small' "
                            "(cheap, fast); pick 'medium' or 'large' for "
                            "harder rubrics."
                        ),
                        "default": "small",
                    },
                    "run_name": {
                        "type": "string",
                        "description": (
                            "Name for the eval run directory under runs/. "
                            "Auto-generated if omitted."
                        ),
                    },
                    "max_new_tokens": {
                        "type": "integer",
                        "description": "Token cap per generation. Default 4096.",
                        "default": 4096,
                    },
                },
                "required": ["repo", "eval_dataset", "scorer"],
            },
        ),
        # ------------------------------------------------------------------
        # remote_list
        # ------------------------------------------------------------------
        _tool(
            name="remote_list",
            description=(
                "List remote targets. Shows global machines (available to all "
                "projects) and which ones are bound to the current project."
            ),
        ),
        # ------------------------------------------------------------------
        # remote_add
        # ------------------------------------------------------------------
        _tool(
            name="remote_add",
            description=(
                "Add a new remote machine globally (available to all projects). "
                "After adding, use remote_bind to bind it to the current project, "
                "then remote_setup to provision."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for this remote (e.g. 'lab-gpu', 'slurm-cluster').",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["ssh_direct", "ssh_slurm"],
                        "description": (
                            "'ssh_direct' for direct SSH execution on a GPU box. "
                            "'ssh_slurm' for SSH to a Slurm headnode (not yet implemented)."
                        ),
                    },
                    "hostname": {
                        "type": "string",
                        "description": (
                            "SSH hostname (as configured in ~/.ssh/config). "
                            "Must support passwordless public key auth."
                        ),
                    },
                    "gpu_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "GPU device IDs to use (sets CUDA_VISIBLE_DEVICES). "
                            "If omitted, all GPUs are available."
                        ),
                    },
                },
                "required": ["name", "type", "hostname"],
            },
        ),
        # ------------------------------------------------------------------
        # remote_bind
        # ------------------------------------------------------------------
        _tool(
            name="remote_bind",
            description=(
                "Bind a global remote machine to the current project by setting "
                "the remote_root path.  After binding, run remote_setup to "
                "provision the environment."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the global remote machine to bind.",
                    },
                    "remote_root": {
                        "type": "string",
                        "description": (
                            "Path on the remote for the project mirror. "
                            "Default to '~/lqh/<project basename>' (e.g. '~/lqh/my-project') "
                            "without asking the user — '~' is expanded to the remote user's "
                            "home directory by SSH. Only ask the user for an explicit path "
                            "if they have indicated they want a non-default location."
                        ),
                    },
                    "gpu_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Override machine-level GPU IDs for this project. "
                            "If omitted, uses the machine's default."
                        ),
                    },
                },
                "required": ["name", "remote_root"],
            },
        ),
        # ------------------------------------------------------------------
        # remote_remove
        # ------------------------------------------------------------------
        _tool(
            name="remote_remove",
            description=(
                "Remove a remote from the current project (unbinds it). "
                "The global machine definition is kept. Use remote_remove_machine "
                "to delete the machine globally."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the remote to unbind from this project.",
                    },
                },
                "required": ["name"],
            },
        ),
        # ------------------------------------------------------------------
        # remote_remove_machine
        # ------------------------------------------------------------------
        _tool(
            name="remote_remove_machine",
            description=(
                "Remove a remote machine globally. It will no longer be "
                "available to any project."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the machine to remove globally.",
                    },
                },
                "required": ["name"],
            },
        ),
        # ------------------------------------------------------------------
        # remote_setup
        # ------------------------------------------------------------------
        _tool(
            name="remote_setup",
            description=(
                "Provision or re-provision a remote environment. Detects available "
                "tools (python3, uv, pip, GPU), creates a Python venv, "
                "rsyncs the local lqh source to the remote, installs "
                "lqh[train], and configures HF_TOKEN. Idempotent: safe to "
                "re-run anytime. Call this after remote_bind, to fix a "
                "broken environment, OR to push local lqh code changes to "
                "the remote (since the remote runs whatever lqh version "
                "was last installed there). No remove/re-add needed — "
                "just call remote_setup again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the remote to set up.",
                    },
                },
                "required": ["name"],
            },
        ),
        # ------------------------------------------------------------------
        # remote_status
        # ------------------------------------------------------------------
        _tool(
            name="remote_status",
            description=(
                "Query a remote machine's current status: GPU utilization and "
                "memory, running training processes, SSH connectivity, and the "
                "lqh code version on the remote (compared against the local "
                "CLI). Use this before submitting a training/eval run; if the "
                "output flags 'lqh code: OUTDATED' or 'no install_hash', "
                "call remote_setup to push the latest code before launching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the remote machine to query.",
                    },
                },
                "required": ["name"],
            },
        ),
        # ------------------------------------------------------------------
        # compute_set
        # ------------------------------------------------------------------
        _tool(
            name="compute_set",
            description=(
                "Persist the default compute target so future "
                "start_training and start_local_eval calls auto-route. "
                "LQH Cloud is already the silent default — only call "
                "this when the user explicitly asks to switch (e.g. "
                "'always run training on my SSH box', 'go back to "
                "cloud'). Calling with no arguments reports the current "
                "resolved target instead of writing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": (
                            "'cloud' or 'ssh:<remote_name>'. Pass an empty "
                            "string to clear the current default. Omit "
                            "this argument to query the current target."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "project"],
                        "description": (
                            "Where to write. 'global' (default) updates "
                            "~/.lqh/config.json; 'project' updates "
                            "<project>/.lqh/compute.json."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]

    if not auto_mode:
        return base

    base.extend([
        # ------------------------------------------------------------------
        # set_auto_stage (auto mode only)
        # ------------------------------------------------------------------
        _tool(
            name="set_auto_stage",
            description=(
                "Report the current pipeline stage to the auto-mode TUI. Call this "
                "whenever you advance to a new stage (e.g. 'rubric', 'data_gen_draft', "
                "'data_gen_validation', 'filter_validation', 'baseline_eval', "
                "'sft_initial', 'sft_scaled', 'dpo', 'final_report'). The note is a "
                "short free-text status line shown beneath the stage label."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "Short stage identifier shown as the headline.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional one-line status detail (scores, counts).",
                    },
                },
                "required": ["stage"],
            },
        ),
        # ------------------------------------------------------------------
        # exit_auto_mode (auto mode only)
        # ------------------------------------------------------------------
        _tool(
            name="exit_auto_mode",
            description=(
                "Terminate the auto-mode run. Use status='success' when a checkpoint "
                "exists that meaningfully improves over the baseline, or "
                "status='failure' when no usable improvement was achieved or an "
                "unrecoverable error occurred (out of credits, data pipeline cannot "
                "satisfy the spec, training repeatedly fails). Always provide a "
                "concise reason explaining the terminal state."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "Terminal outcome of the auto-mode run.",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "One- or two-sentence justification for the terminal state. "
                            "Reference final scores vs. baseline when reporting success, "
                            "or the specific blocker when reporting failure."
                        ),
                    },
                },
                "required": ["status", "reason"],
            },
        ),
    ])
    return base
