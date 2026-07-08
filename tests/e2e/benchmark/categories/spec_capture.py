"""Category 1: Spec Capture benchmark scenarios.

Tests whether the LLM asks good follow-up questions to clarify ambiguous
requirements before creating SPEC.md.
"""

from __future__ import annotations

from tests.e2e.scenarios import Scenario

SPEC_CAPTURE_AMBIGUOUS_OUTPUT = Scenario(
    name="bench_spec_capture_ambiguous_output",
    description=(
        "You are a user who wants to build a model that answers questions about "
        "your product documentation. You have NOT specified the output format "
        "(prose, JSON, bullet points, etc.), the expected length of answers, "
        "or how the model should handle questions it cannot answer.\n\n"
        "Behavior rules:\n"
        "- Give short, direct answers to clarifying questions\n"
        "- When asked about output format, say 'JSON with answer and confidence'\n"
        "- When asked about unanswerable questions, say 'respond with a message "
        "saying the info is not in the docs'\n"
        "- When asked about doc format, say 'markdown files, typically 1-5 pages'\n"
        "- When asked about examples, let the agent generate them and say 'looks good'\n"
        "- When asked about anything already stated, say 'I already mentioned that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message=(
        "I want to build a model that answers questions about my product documentation"
    ),
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for how well it captures a documentation Q&A task.\n"
        "Check for:\n"
        "- Output format described (JSON with answer and confidence)\n"
        "- Handling of unanswerable questions described\n"
        "- Input format described (product documentation in markdown)\n"
        "- Clear and actionable specification\n\n"
        "10 = all requirements captured precisely\n"
        "5 = some requirements missing or vague\n"
        "1 = does not describe the task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_AMBIGUOUS_LANGS = Scenario(
    name="bench_spec_capture_ambiguous_langs",
    description=(
        "You are a user who wants to build a multilingual chatbot. You have NOT "
        "specified which languages, whether to auto-detect the input language, "
        "or whether to translate between languages or respond in the user's language.\n\n"
        "Behavior rules:\n"
        "- When asked about languages, say 'English, Spanish, and Mandarin'\n"
        "- When asked about language detection, say 'auto-detect the user's language "
        "and respond in the same language'\n"
        "- When asked about translation, say 'no translation, just respond in "
        "whatever language the user writes in'\n"
        "- When asked about tone, say 'friendly and professional'\n"
        "- When asked about domain, say 'customer support for a SaaS product'\n"
        "- When asked about examples, let the agent generate them\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message=(
        "I need a multilingual chatbot"
    ),
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a multilingual chatbot task.\n"
        "Check for:\n"
        "- Languages listed (English, Spanish, Mandarin)\n"
        "- Language detection mentioned (auto-detect)\n"
        "- Response language behavior (respond in user's language)\n"
        "- Domain specified (customer support / SaaS)\n\n"
        "10 = all captured, 5 = some missing, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_ALREADY_DETAILED = Scenario(
    name="bench_spec_capture_already_detailed",
    description=(
        "You are a user who has already thought everything through. Your initial "
        "message contains all requirements. You get mildly annoyed if the agent "
        "asks questions about things you already stated.\n\n"
        "Behavior rules:\n"
        "- If the agent asks about something already in your initial message, say "
        "'I already specified that in my first message'\n"
        "- If the agent asks a genuinely new question not covered in the initial "
        "message, answer helpfully\n"
        "- When asked about examples, say 'the examples in my message should be enough'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message=(
        "I need a model that classifies customer support tickets into categories. "
        "Here are the exact requirements:\n\n"
        "Input: plain text customer support ticket, 1-10 sentences\n"
        "Output: JSON with 'category' (one of: billing, technical, account, shipping, "
        "general) and 'priority' (low, medium, high, urgent)\n\n"
        "Rules:\n"
        "- If the ticket mentions money, refund, or charge, it's 'billing'\n"
        "- If it mentions error, crash, or bug, it's 'technical'\n"
        "- If it mentions password, login, or access, it's 'account'\n"
        "- If it mentions delivery, tracking, or package, it's 'shipping'\n"
        "- Everything else is 'general'\n"
        "- Priority is 'urgent' if the customer mentions legal action or regulatory issues\n"
        "- Priority is 'high' if the customer is clearly frustrated or it's a business-critical issue\n"
        "- Priority is 'medium' for standard requests\n"
        "- Priority is 'low' for informational questions\n\n"
        "Example:\n"
        "Input: 'I was charged twice for my subscription last month and I want a refund immediately'\n"
        "Output: {\"category\": \"billing\", \"priority\": \"high\"}\n\n"
        "Please create the SPEC.md."
    ),
    expected_tools=["create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a ticket classification task.\n"
        "Check for:\n"
        "- All 5 categories listed (billing, technical, account, shipping, general)\n"
        "- All 4 priority levels listed (low, medium, high, urgent)\n"
        "- Classification rules captured\n"
        "- JSON output format with both fields\n"
        "- Example included\n\n"
        "10 = everything from initial message captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=15,
    stage_limits={"spec_capture": 12},
)


SPEC_CAPTURE_AMBIGUOUS_SUMMARIZATION = Scenario(
    name="bench_spec_capture_ambiguous_summarization",
    description=(
        "You are a user who wants a model that summarizes documents. You have NOT "
        "specified the summary length, the style (extractive vs abstractive), the "
        "output format, or what kinds of documents.\n\n"
        "Behavior rules:\n"
        "- When asked about document type, say 'long news and research articles, "
        "1000-8000 words'\n"
        "- When asked about summary length, say 'about 3-5 sentences, never more "
        "than 120 words'\n"
        "- When asked extractive vs abstractive, say 'abstractive, written in our "
        "own words'\n"
        "- When asked about output format, say 'JSON with a summary string and a "
        "list of 3 key_points'\n"
        "- When asked about language, say 'English only'\n"
        "- When asked about examples, let the agent generate them and say 'looks good'\n"
        "- When asked about anything already stated, say 'I already mentioned that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message="I want a model that summarizes documents for me",
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a document summarization task.\n"
        "Check for:\n"
        "- Summary length bound captured (~3-5 sentences / <=120 words)\n"
        "- Abstractive style specified\n"
        "- Output format (JSON with summary + key_points list)\n"
        "- Input document type/length described\n\n"
        "10 = all captured precisely, 5 = some missing/vague, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_AMBIGUOUS_PII = Scenario(
    name="bench_spec_capture_ambiguous_pii",
    description=(
        "You are a user who wants a model that redacts PII from text. You have NOT "
        "specified which entity types to redact, the masking style, or whether the "
        "original values must be recoverable.\n\n"
        "Behavior rules:\n"
        "- When asked which entities, say 'names, emails, phone numbers, credit "
        "card numbers, and street addresses'\n"
        "- When asked about masking style, say 'replace each with a typed "
        "placeholder like [NAME], [EMAIL], [PHONE]'\n"
        "- When asked whether redaction must be reversible, say 'no, it must NOT "
        "be reversible — do not keep a mapping'\n"
        "- When asked about input, say 'free-form customer support chat logs, "
        "English'\n"
        "- When asked about output format, say 'return the redacted text as a "
        "plain string'\n"
        "- When asked about examples, let the agent generate them\n"
        "- When asked about anything already stated, say 'I already said that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message="I need a model that strips personal information out of text",
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a PII redaction task.\n"
        "Check for:\n"
        "- The 5 entity types listed (names, emails, phones, credit cards, addresses)\n"
        "- Typed-placeholder masking style ([NAME], [EMAIL], ...)\n"
        "- Non-reversible requirement (no stored mapping)\n"
        "- Input/output format (chat logs in, redacted string out)\n\n"
        "10 = all captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_AMBIGUOUS_CODE_REVIEW = Scenario(
    name="bench_spec_capture_ambiguous_code_review",
    description=(
        "You are a user who wants a model that reviews code diffs and leaves "
        "comments. You have NOT specified which languages, what the comments "
        "should focus on, the severity scheme, or the output format.\n\n"
        "Behavior rules:\n"
        "- When asked which languages, say 'Python and TypeScript only'\n"
        "- When asked what to focus on, say 'correctness bugs and security issues "
        "— not style or formatting'\n"
        "- When asked about severity, say 'each finding gets a severity: low, "
        "medium, high, or critical'\n"
        "- When asked about output format, say 'a JSON list of findings, each "
        "with file, line, severity, and message'\n"
        "- When asked what the input looks like, say 'a unified diff of a pull "
        "request'\n"
        "- When asked about false positives, say 'prefer precision — only report "
        "things you are confident about'\n"
        "- When asked about examples, let the agent generate them\n"
        "- When asked about anything already stated, say 'I already covered that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message="I want a model that does automated code review on pull requests",
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for an automated code-review task.\n"
        "Check for:\n"
        "- Languages scoped (Python, TypeScript)\n"
        "- Focus on correctness + security (not style)\n"
        "- Severity scheme (low/medium/high/critical)\n"
        "- Output format (JSON list with file/line/severity/message)\n"
        "- Input is a unified diff\n\n"
        "10 = all captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_AMBIGUOUS_ASPECT = Scenario(
    name="bench_spec_capture_ambiguous_aspect_sentiment",
    description=(
        "You are a user who wants aspect-based sentiment analysis of product "
        "reviews. You have NOT specified which aspects, the sentiment scale, how "
        "to handle aspects not mentioned in a review, or the output format.\n\n"
        "Behavior rules:\n"
        "- When asked which aspects, say 'price, quality, shipping, and customer "
        "service'\n"
        "- When asked about the sentiment scale, say 'positive, negative, or "
        "neutral per aspect'\n"
        "- When asked about aspects not mentioned in a review, say 'mark them as "
        "not_mentioned, do not guess'\n"
        "- When asked about output format, say 'JSON object mapping each aspect to "
        "its sentiment'\n"
        "- When asked about input, say 'English product reviews, 1-10 sentences'\n"
        "- When asked about examples, let the agent generate them\n"
        "- When asked about anything already stated, say 'I already told you that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message="I want to analyze the sentiment of product reviews by aspect",
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for an aspect-based sentiment task.\n"
        "Check for:\n"
        "- The 4 aspects listed (price, quality, shipping, customer service)\n"
        "- Per-aspect sentiment scale (positive/negative/neutral)\n"
        "- not_mentioned handling for absent aspects\n"
        "- Output format (JSON aspect -> sentiment)\n\n"
        "10 = all captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_AMBIGUOUS_SQL = Scenario(
    name="bench_spec_capture_ambiguous_sql",
    description=(
        "You are a user who wants a text-to-SQL model. You have NOT specified the "
        "SQL dialect, whether the schema is provided at inference time, how to "
        "handle unanswerable questions, or the output format.\n\n"
        "Behavior rules:\n"
        "- When asked about the SQL dialect, say 'PostgreSQL'\n"
        "- When asked about the schema, say 'the table schema is given in the "
        "prompt alongside each question'\n"
        "- When asked about read vs write, say 'SELECT queries only, never "
        "mutate data'\n"
        "- When asked about unanswerable questions, say 'if the question cannot be "
        "answered from the schema, return the string UNANSWERABLE'\n"
        "- When asked about output format, say 'just the raw SQL string, no "
        "markdown fences'\n"
        "- When asked about examples, let the agent generate them\n"
        "- When asked about anything already stated, say 'I already said that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message="I want a model that turns natural-language questions into SQL",
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a text-to-SQL task.\n"
        "Check for:\n"
        "- Dialect specified (PostgreSQL)\n"
        "- Schema supplied at inference time (in prompt)\n"
        "- SELECT-only / read-only constraint\n"
        "- UNANSWERABLE handling\n"
        "- Output format (raw SQL string, no fences)\n\n"
        "10 = all captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_AMBIGUOUS_INTENT = Scenario(
    name="bench_spec_capture_ambiguous_intent",
    description=(
        "You are a user who wants an intent + slot model for a smart-home voice "
        "assistant. You have NOT specified the intent set, the slots, what to do "
        "with out-of-scope utterances, or the output format.\n\n"
        "Behavior rules:\n"
        "- When asked about intents, say 'set_temperature, turn_on_device, "
        "turn_off_device, and play_music'\n"
        "- When asked about slots, say 'device name, room, temperature value, and "
        "song/artist — only the ones relevant to the intent'\n"
        "- When asked about out-of-scope utterances, say 'return intent "
        "out_of_scope with empty slots'\n"
        "- When asked about output format, say 'JSON with intent and a slots "
        "object'\n"
        "- When asked about language, say 'English'\n"
        "- When asked about examples, let the agent generate them\n"
        "- When asked about anything already stated, say 'I already mentioned that'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message="I need an intent classifier for a smart-home assistant",
    expected_tools=["ask_user", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for an intent + slot-filling task.\n"
        "Check for:\n"
        "- The 4 intents listed (set_temperature, turn_on_device, turn_off_device, play_music)\n"
        "- Slots described (device, room, temperature, song/artist)\n"
        "- out_of_scope handling\n"
        "- Output format (JSON with intent + slots)\n\n"
        "10 = all captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=25,
    stage_limits={"spec_capture": 20},
)

SPEC_CAPTURE_ALREADY_DETAILED_NER = Scenario(
    name="bench_spec_capture_already_detailed_ner",
    description=(
        "You are a user who has fully specified a named-entity-recognition task in "
        "your first message. You get mildly annoyed if the agent re-asks anything "
        "you already stated.\n\n"
        "Behavior rules:\n"
        "- If the agent asks about something already in your initial message, say "
        "'I already specified that in my first message'\n"
        "- If the agent asks a genuinely new question not covered above, answer "
        "helpfully\n"
        "- When asked about examples, say 'the example in my message is enough'\n"
        "- When offered next steps after spec creation, say 'I'm done for now'"
    ),
    initial_message=(
        "I need a named-entity-recognition model. Full requirements:\n\n"
        "Input: a plain-text English sentence or paragraph, up to 512 words.\n"
        "Entities to extract: PERSON, ORG, LOCATION, DATE, MONEY.\n"
        "Output: a JSON list of objects, each with 'text' (the exact span), "
        "'type' (one of the 5 labels), and 'start' (character offset).\n\n"
        "Rules:\n"
        "- Spans must be exact substrings of the input (no normalization).\n"
        "- Overlapping entities are not allowed; prefer the longer span.\n"
        "- Nationalities and adjectives (e.g. 'French') are NOT locations.\n"
        "- Relative dates ('yesterday') ARE dates.\n"
        "- If no entities are present, return an empty list [].\n\n"
        "Example:\n"
        "Input: 'Tim Cook visited Berlin in March 2024.'\n"
        "Output: [{\"text\": \"Tim Cook\", \"type\": \"PERSON\", \"start\": 0}, "
        "{\"text\": \"Berlin\", \"type\": \"LOCATION\", \"start\": 17}, "
        "{\"text\": \"March 2024\", \"type\": \"DATE\", \"start\": 27}]\n\n"
        "Please create the SPEC.md."
    ),
    expected_tools=["create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Score the SPEC.md for a named-entity-recognition task.\n"
        "Check for:\n"
        "- All 5 entity types (PERSON, ORG, LOCATION, DATE, MONEY)\n"
        "- Output format (JSON list with text/type/start)\n"
        "- Exact-substring and no-overlap rules\n"
        "- Edge rules (nationalities not locations; relative dates are dates; "
        "empty list when none)\n"
        "- Example included\n\n"
        "10 = everything from initial message captured, 5 = partial, 1 = wrong task"
    ),
    max_turns=15,
    stage_limits={"spec_capture": 12},
)


SCENARIOS = [
    SPEC_CAPTURE_AMBIGUOUS_OUTPUT,
    SPEC_CAPTURE_AMBIGUOUS_LANGS,
    SPEC_CAPTURE_ALREADY_DETAILED,
    SPEC_CAPTURE_AMBIGUOUS_SUMMARIZATION,
    SPEC_CAPTURE_AMBIGUOUS_PII,
    SPEC_CAPTURE_AMBIGUOUS_CODE_REVIEW,
    SPEC_CAPTURE_AMBIGUOUS_ASPECT,
    SPEC_CAPTURE_AMBIGUOUS_SQL,
    SPEC_CAPTURE_AMBIGUOUS_INTENT,
    SPEC_CAPTURE_ALREADY_DETAILED_NER,
]
