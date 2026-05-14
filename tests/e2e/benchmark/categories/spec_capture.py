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


SCENARIOS = [
    SPEC_CAPTURE_AMBIGUOUS_OUTPUT,
    SPEC_CAPTURE_AMBIGUOUS_LANGS,
    SPEC_CAPTURE_ALREADY_DETAILED,
]
