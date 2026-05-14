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
                "Do NOT include an 'Other' option - the TUI always appends one automatically. "
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
                            "the model outputs, and writes to evals/runs/<run_name>/."
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
                            "Model to use for inference in mode='model_eval'. "
                            "Use list_models to discover available LFM models. "
                            "Examples: 'lfm2.5-1.2b-instruct', 'small', 'medium', "
                            "'random:small:seed123'. Required for mode='model_eval'."
                        ),
                    },
                    "inference_system_prompt": {
                        "type": "string",
                        "description": (
                            "System prompt to prepend to each sample before running "
                            "inference. Used in mode='model_eval' to test different "
                            "prompt strategies with the same model and eval dataset."
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
                "List available Liquid Foundation Models (LFMs) from the API. "
                "Returns model IDs, display names, HuggingFace checkpoint IDs, "
                "and context lengths. Use this to discover which models are "
                "available for inference in model evaluation runs."
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
                "Push a local dataset to Hugging Face Hub. Creates the repo if it "
                "does not exist (private by default). Requires HF_TOKEN env var. "
                "User permission is requested before pushing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "local_path": {
                        "type": "string",
                        "description": (
                            "Relative path to the local dataset directory "
                            "(e.g. 'datasets/summarization_v1'). Must contain "
                            "data.parquet."
                        ),
                    },
                    "repo_id": {
                        "type": "string",
                        "description": (
                            "HF repo ID (e.g. 'username/my-dataset'). If omitted, "
                            "auto-generated from HF username + project dir + dataset "
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
                            "Defaults to 'train'."
                        ),
                        "default": "train",
                    },
                    "subset": {
                        "type": "string",
                        "description": (
                            "Dataset config/subset name. Use this to push multiple "
                            "related datasets as subsets of a single HF repo."
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
                "Download a dataset from Hugging Face Hub to local storage. "
                "Saves as parquet in datasets/. Passes HF_TOKEN if available "
                "for private repos."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": (
                            "HF dataset repo ID (e.g. 'username/my-dataset')."
                        ),
                    },
                    "local_path": {
                        "type": "string",
                        "description": (
                            "Where to save locally, relative to project dir. "
                            "Defaults to 'datasets/{repo-name}/'."
                        ),
                    },
                    "split": {
                        "type": "string",
                        "description": (
                            "Specific split to download (e.g. 'train'). "
                            "If omitted, downloads all splits."
                        ),
                    },
                    "subset": {
                        "type": "string",
                        "description": (
                            "Dataset config/subset name. Required if the dataset "
                            "has multiple configs."
                        ),
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Specific files to download instead of the full dataset "
                            "(e.g. ['data/train-00000-of-00001.parquet']). "
                            "Useful for downloading single parquet files."
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
                "(pip install lqh[train]). User permission is requested before starting."
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
                        "type": "string",
                        "description": (
                            "Relative path to the training dataset directory "
                            "(e.g. 'datasets/summarization_v1'). Must contain data.parquet."
                        ),
                    },
                    "eval_dataset": {
                        "type": "string",
                        "description": (
                            "Relative path to the eval dataset directory. If provided, "
                            "checkpoint evaluations are run automatically via the API judge."
                        ),
                    },
                    "scorer": {
                        "type": "string",
                        "description": (
                            "Relative path to the scorer .md file for checkpoint evaluations "
                            "(e.g. 'evals/scorers/summarization_v1.md')."
                        ),
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
                    "num_epochs": {
                        "type": "integer",
                        "description": "Number of training epochs (SFT only). Default: 3.",
                        "default": 3,
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate. Default: 2e-5 for SFT, 5e-6 for DPO.",
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
                    "remote": {
                        "type": "string",
                        "description": (
                            "Name of a configured remote to run training on "
                            "(e.g. 'lab-gpu'). If omitted, runs locally. "
                            "Use remote_list to see configured remotes."
                        ),
                    },
                },
                "required": ["type", "base_model", "dataset"],
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
                "If run_name is omitted, shows status of all runs."
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
                    "remote": {
                        "type": "string",
                        "description": (
                            "Name of the remote where the run is executing. "
                            "If omitted, checks local runs."
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
                    "remote": {
                        "type": "string",
                        "description": (
                            "Name of the remote where the run is executing. "
                            "If omitted, stops a local run."
                        ),
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
                "Run local model inference as a subprocess and score the results "
                "via the API judge. Use this to evaluate a locally-stored model "
                "(e.g. a training checkpoint) on an eval dataset without using "
                "the API for inference. Requires the 'train' optional dependencies. "
                "Output lives at runs/<run_name>/ — predictions.parquet plus "
                "eval_result.json once scoring finishes (NOT under evals/runs/, "
                "which is for run_scoring's API-mode evals)."
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
                        "type": "string",
                        "description": (
                            "Relative path to the eval dataset directory "
                            "(must contain data.parquet)."
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
                    "remote": {
                        "type": "string",
                        "description": (
                            "Name of a configured remote to run inference on. "
                            "If omitted, runs locally."
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
