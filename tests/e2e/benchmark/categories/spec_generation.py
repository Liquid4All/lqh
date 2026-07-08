"""Category 2: Spec Generation benchmark scenarios.

Tests whether the LLM can synthesize a SPEC.md from a long conversation
history without missing any requirements.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.scenarios import Scenario


# ---------------------------------------------------------------------------
# Planted conversation: NLP classification task (15 turns)
# Each requirement is annotated with [REQ-N] in comments for tracking
# ---------------------------------------------------------------------------

_NLP_CONVERSATION = """\
# Prior Conversation Transcript

**Turn 1 - User**: I want to build a model that classifies news articles by topic.

**Turn 2 - Agent**: What topic categories do you need?

**Turn 3 - User**: Politics, Technology, Sports, Entertainment, and Science. Oh and also Business, I almost forgot that one.

**Turn 4 - Agent**: Got it, 6 categories. What's the input format?

**Turn 5 - User**: The input is the full article text, but we should also accept just a headline if that's all the user has. So it should work with both short and long text.

**Turn 6 - Agent**: Makes sense. What about the output format?

**Turn 7 - User**: JSON with the predicted category and a confidence score between 0 and 1. Like {"category": "Technology", "confidence": 0.92}

**Turn 8 - Agent**: Should the model handle articles in languages other than English?

**Turn 9 - User**: Good question - yes, it should handle English and Spanish articles. But the category labels should always be in English regardless of the input language.

**Turn 10 - Agent**: What about articles that could fit multiple categories?

**Turn 11 - User**: Return the single best category. But (this is important) if the confidence is below 0.5, return "Unknown" as the category instead of guessing.

**Turn 12 - Agent**: Any special handling for edge cases?

**Turn 13 - User**: Yes - if the article is about a tech company's stock price, it should be classified as "Business" not "Technology". Financial topics always take precedence. Also, opinion pieces should be classified by their subject matter, not as a separate category.

**Turn 14 - Agent**: What about article length limits?

**Turn 15 - User**: Articles can be up to 5000 words, but the model should work well even with very short inputs like a single sentence headline. No minimum length requirement.
"""

# Requirements planted in the conversation:
# REQ-1: 6 categories (Politics, Technology, Sports, Entertainment, Science, Business)
# REQ-2: Input is full article text OR headline (short and long)
# REQ-3: JSON output with category and confidence (0-1)
# REQ-4: Handles English and Spanish
# REQ-5: Category labels always in English
# REQ-6: If confidence < 0.5, return "Unknown"
# REQ-7: Tech company stock -> Business (financial precedence rule)
# REQ-8: Opinion pieces classified by subject matter
# REQ-9: Up to 5000 words, no minimum length

SPEC_GEN_NLP_TASK = Scenario(
    name="bench_spec_gen_nlp_task",
    description=(
        "You are a user who already had a long conversation about a news classification "
        "task. The conversation transcript is in prior_conversation.md. You want the "
        "agent to create SPEC.md from it.\n\n"
        "Behavior rules:\n"
        "- If the agent asks to clarify something from the conversation, answer briefly\n"
        "- If the agent asks about something NOT in the conversation, say 'that's not "
        "needed, just use what's in the conversation'\n"
        "- When shown the SPEC.md, check it briefly and say 'looks good' or point out "
        "if something obvious is missing\n"
        "- When offered next steps, say 'I'm done for now'"
    ),
    initial_message=(
        "I've been discussing my news classification task in a prior conversation. "
        "The transcript is at prior_conversation.md. Please read it and create the "
        "SPEC.md based on everything we discussed."
    ),
    expected_tools=["read_file", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Check the SPEC.md against these specific requirements from the conversation:\n"
        "1. 6 categories: Politics, Technology, Sports, Entertainment, Science, Business\n"
        "2. Input accepts both full articles and headlines\n"
        "3. JSON output with 'category' and 'confidence' (0-1)\n"
        "4. Handles English and Spanish input\n"
        "5. Category labels always in English\n"
        "6. Confidence < 0.5 returns 'Unknown' category\n"
        "7. Financial topics take precedence (tech stock -> Business)\n"
        "8. Opinion pieces classified by subject matter\n"
        "9. Up to 5000 words input, no minimum\n\n"
        "Score: count how many of these 9 requirements are captured.\n"
        "10 = all 9, 7 = 6-7 requirements, 5 = 4-5, 3 = 2-3, 1 = 0-1"
    ),
    max_turns=15,
    stage_limits={"spec_capture": 12},
    seed_fn=lambda project_dir: Path(project_dir / "prior_conversation.md").write_text(
        _NLP_CONVERSATION, encoding="utf-8"
    ),
)


# ---------------------------------------------------------------------------
# Planted conversation: Tool-calling task
# ---------------------------------------------------------------------------

_TOOL_CONVERSATION = """\
# Prior Conversation Transcript

**Turn 1 - User**: I want to build an AI assistant that helps with project management using API tools.

**Turn 2 - Agent**: What tools should the assistant have access to?

**Turn 3 - User**: Three tools:
1. `list_tasks` - takes a project_id (string) and optional status filter ("open", "closed", "all"), returns a list of task objects
2. `create_task` - takes project_id, title (string), description (string), assignee (string, optional), due_date (string in YYYY-MM-DD format, optional)
3. `update_task` - takes task_id and any subset of: title, description, status ("open"/"in_progress"/"closed"), assignee, due_date

**Turn 4 - Agent**: What should the assistant do after calling a tool?

**Turn 5 - User**: After getting tool results, it should summarize the outcome in natural language. For list_tasks, give a brief overview. For create/update, confirm what was done.

**Turn 6 - Agent**: How should it handle errors from the tools?

**Turn 7 - User**: If a tool returns an error, the assistant should explain what went wrong in simple terms and suggest what the user can do. Never expose raw error messages or stack traces.

**Turn 8 - Agent**: What if the user's request is ambiguous?

**Turn 9 - User**: Ask for clarification. For example if they say "mark it done" but haven't specified which task, ask them to clarify. But if context from earlier in the conversation makes it obvious, just proceed.

**Turn 10 - Agent**: Should the assistant ever call multiple tools in sequence?

**Turn 11 - User**: Yes! For example, if someone says "create a task and assign it to Sarah", that's one create_task call. But if they say "show me all open tasks and close the overdue ones", that requires list_tasks first, then potentially multiple update_task calls. The assistant should handle multi-step workflows.

**Turn 12 - Agent**: Any constraints on the assistant's behavior?

**Turn 13 - User**: It should never modify tasks without the user's intent - no auto-closing or auto-assigning. Also, it should always confirm before doing bulk operations (more than 3 updates at once). And one more thing - the assistant should recognize when a request is outside its capabilities and say so politely.
"""

# Requirements:
# REQ-1: 3 tools: list_tasks, create_task, update_task with exact parameters
# REQ-2: list_tasks params: project_id, optional status filter
# REQ-3: create_task params: project_id, title, description, optional assignee, optional due_date (YYYY-MM-DD)
# REQ-4: update_task params: task_id, optional subset of title/description/status/assignee/due_date
# REQ-5: Summarize tool results in natural language
# REQ-6: Handle tool errors gracefully, no raw error messages
# REQ-7: Ask for clarification on ambiguous requests, but use conversation context
# REQ-8: Handle multi-step workflows (sequential tool calls)
# REQ-9: Never modify without user intent, confirm bulk operations (>3)
# REQ-10: Recognize out-of-scope requests

SPEC_GEN_TOOL_CALLING = Scenario(
    name="bench_spec_gen_tool_calling",
    description=(
        "You are a user who discussed a project management tool-calling assistant. "
        "The conversation transcript is in prior_conversation.md. You want the "
        "agent to create SPEC.md from it.\n\n"
        "Behavior rules:\n"
        "- If the agent asks to clarify, answer briefly\n"
        "- If the agent asks about things not in the conversation, say 'stick to "
        "what we discussed'\n"
        "- When shown SPEC.md, say 'looks good'\n"
        "- When offered next steps, say 'I'm done for now'"
    ),
    initial_message=(
        "The conversation transcript for my project management assistant is at "
        "prior_conversation.md. Please read it and create SPEC.md."
    ),
    expected_tools=["read_file", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Check the SPEC.md against these requirements from the conversation:\n"
        "1. Three tools defined: list_tasks, create_task, update_task\n"
        "2. list_tasks parameters: project_id, optional status filter\n"
        "3. create_task parameters: project_id, title, description, optional assignee, optional due_date (YYYY-MM-DD)\n"
        "4. update_task parameters: task_id, optional fields (title/description/status/assignee/due_date)\n"
        "5. Summarize tool results in natural language\n"
        "6. Handle tool errors gracefully\n"
        "7. Clarify ambiguous requests (use conversation context)\n"
        "8. Support multi-step workflows (sequential tool calls)\n"
        "9. Never modify without user intent, confirm bulk operations (>3)\n"
        "10. Recognize out-of-scope requests\n\n"
        "Score: count captured out of 10.\n"
        "10 = all 10, 7 = 7, 5 = 5, 3 = 3, 1 = 0-1"
    ),
    max_turns=15,
    stage_limits={"spec_capture": 12},
    seed_fn=lambda project_dir: Path(project_dir / "prior_conversation.md").write_text(
        _TOOL_CONVERSATION, encoding="utf-8"
    ),
)


# ---------------------------------------------------------------------------
# Planted conversation: Summarization task
# ---------------------------------------------------------------------------

_SUMMARIZATION_CONVERSATION = """\
# Prior Conversation Transcript

**Turn 1 - User**: I want a model that summarizes customer support email threads for our agents.

**Turn 2 - Agent**: What should the summary contain?

**Turn 3 - User**: Three things: a one-sentence TL;DR, the customer's main issue, and whether the issue is resolved or still open.

**Turn 4 - Agent**: What's the input?

**Turn 5 - User**: The full email thread - could be 2 to 20 messages back and forth. Each message is labeled with who sent it (customer or agent).

**Turn 6 - Agent**: What output format do you want?

**Turn 7 - User**: JSON with keys: tldr (string), issue (string), status (one of "resolved", "open", "needs_followup"). Status must be exactly one of those three values.

**Turn 8 - Agent**: How long should the TL;DR and issue fields be?

**Turn 9 - User**: TL;DR is one sentence, max 25 words. The issue description can be up to 3 sentences. Keep it factual, no marketing language.

**Turn 10 - Agent**: Should it handle threads in other languages?

**Turn 11 - User**: Yes, the thread could be in English or German, but the summary itself should always be written in English.

**Turn 12 - Agent**: Any edge cases to handle?

**Turn 13 - User**: A couple. If the thread is just spam or has no real support issue, set issue to "no_issue" and status to "resolved". And never include any personal data like full credit card numbers in the summary - mask them if they appear.

**Turn 14 - Agent**: Anything about tone or audience?

**Turn 15 - User**: The reader is an internal support agent, so be concise and direct. No need for greetings or sign-offs in the summary.
"""

# Requirements:
# REQ-1: Summary contains TL;DR, main issue, and resolution status
# REQ-2: Input is a labeled multi-message email thread (2-20 messages)
# REQ-3: JSON output: tldr, issue, status
# REQ-4: status is exactly one of resolved/open/needs_followup
# REQ-5: TL;DR <= 25 words / one sentence; issue <= 3 sentences; factual
# REQ-6: Input English or German; summary always in English
# REQ-7: Spam/no-issue -> issue="no_issue", status="resolved"
# REQ-8: Mask personal data (e.g. credit card numbers)
# REQ-9: Concise internal-agent tone, no greetings/sign-offs

SPEC_GEN_SUMMARIZATION = Scenario(
    name="bench_spec_gen_summarization",
    description=(
        "You are a user who already discussed a support-email summarization task. "
        "The transcript is in prior_conversation.md. You want the agent to create "
        "SPEC.md from it.\n\n"
        "Behavior rules:\n"
        "- If the agent asks to clarify something from the conversation, answer briefly\n"
        "- If the agent asks about something NOT in the conversation, say 'that's not "
        "needed, just use what's in the conversation'\n"
        "- When shown the SPEC.md, check it briefly and say 'looks good' or point out "
        "if something obvious is missing\n"
        "- When offered next steps, say 'I'm done for now'"
    ),
    initial_message=(
        "I've been discussing a support-email summarization task. The transcript is "
        "at prior_conversation.md. Please read it and create SPEC.md based on "
        "everything we discussed."
    ),
    expected_tools=["read_file", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Check the SPEC.md against these requirements from the conversation:\n"
        "1. Summary contains TL;DR, main issue, and resolution status\n"
        "2. Input is a labeled multi-message email thread (2-20 messages)\n"
        "3. JSON output with keys tldr, issue, status\n"
        "4. status is exactly one of resolved/open/needs_followup\n"
        "5. TL;DR <= 25 words (one sentence); issue <= 3 sentences; factual\n"
        "6. Input may be English or German; summary always in English\n"
        "7. Spam/no real issue -> issue='no_issue', status='resolved'\n"
        "8. Mask personal data such as credit card numbers\n"
        "9. Concise internal-agent tone, no greetings/sign-offs\n\n"
        "Score: count how many of these 9 requirements are captured.\n"
        "10 = all 9, 7 = 6-7, 5 = 4-5, 3 = 2-3, 1 = 0-1"
    ),
    max_turns=15,
    stage_limits={"spec_capture": 12},
    seed_fn=lambda project_dir: Path(project_dir / "prior_conversation.md").write_text(
        _SUMMARIZATION_CONVERSATION, encoding="utf-8"
    ),
)


# ---------------------------------------------------------------------------
# Planted conversation: Content moderation task
# ---------------------------------------------------------------------------

_MODERATION_CONVERSATION = """\
# Prior Conversation Transcript

**Turn 1 - User**: We need a content moderation model for user-generated comments on our forum.

**Turn 2 - Agent**: What categories of harmful content should it detect?

**Turn 3 - User**: Four: harassment, hate_speech, sexual_content, and self_harm. A comment can match more than one.

**Turn 4 - Agent**: So it's multi-label?

**Turn 5 - User**: Yes, multi-label. Output a JSON object with each of the four categories mapped to true or false. All four keys must always be present.

**Turn 6 - Agent**: Do you also need an overall action?

**Turn 7 - User**: Good point - yes. Add an "action" field that is one of: allow, flag_for_review, or remove. If self_harm is true, the action must always be flag_for_review regardless of anything else, because we want a human to reach out.

**Turn 8 - Agent**: What about borderline cases?

**Turn 9 - User**: Mild profanity on its own is NOT harassment or hate_speech - allow it. Only flag profanity when it's directed at a person or group. And quoting someone else's hateful comment to criticize it should not be labeled hate_speech.

**Turn 10 - Agent**: What languages?

**Turn 11 - User**: English and Spanish comments. The category labels stay in English.

**Turn 12 - Agent**: Any input constraints?

**Turn 13 - User**: Comments are short, up to 1000 characters. If a comment is empty or just whitespace, return all four categories false and action allow.

**Turn 14 - Agent**: Should it explain its decisions?

**Turn 15 - User**: Yes - include a short "reason" string (max one sentence) explaining the most severe label, or "clean" if nothing was flagged.
"""

# Requirements:
# REQ-1: 4 categories: harassment, hate_speech, sexual_content, self_harm
# REQ-2: Multi-label; all four keys always present (bool each)
# REQ-3: "action" field: allow / flag_for_review / remove
# REQ-4: self_harm true -> action must be flag_for_review
# REQ-5: Mild standalone profanity is allowed; only directed profanity flagged
# REQ-6: Quoting hateful content to criticize is not hate_speech
# REQ-7: English + Spanish input; labels in English
# REQ-8: Up to 1000 chars; empty/whitespace -> all false + action allow
# REQ-9: "reason" string (<= one sentence) or "clean"

SPEC_GEN_MODERATION = Scenario(
    name="bench_spec_gen_moderation",
    description=(
        "You are a user who already discussed a content-moderation task. The "
        "transcript is in prior_conversation.md. You want the agent to create "
        "SPEC.md from it.\n\n"
        "Behavior rules:\n"
        "- If the agent asks to clarify something from the conversation, answer briefly\n"
        "- If the agent asks about something NOT in the conversation, say 'stick to "
        "what we discussed'\n"
        "- When shown the SPEC.md, say 'looks good' or point out an obvious gap\n"
        "- When offered next steps, say 'I'm done for now'"
    ),
    initial_message=(
        "The conversation transcript for my content-moderation model is at "
        "prior_conversation.md. Please read it and create SPEC.md."
    ),
    expected_tools=["read_file", "create_file"],
    expected_files=["SPEC.md"],
    judge_criteria=(
        "Check the SPEC.md against these requirements from the conversation:\n"
        "1. Four categories: harassment, hate_speech, sexual_content, self_harm\n"
        "2. Multi-label; all four keys always present as booleans\n"
        "3. 'action' field: allow / flag_for_review / remove\n"
        "4. self_harm true forces action=flag_for_review\n"
        "5. Mild standalone profanity allowed; only directed profanity flagged\n"
        "6. Quoting hateful content to criticize is not hate_speech\n"
        "7. English + Spanish input; labels stay in English\n"
        "8. <=1000 chars; empty/whitespace -> all false + action allow\n"
        "9. 'reason' string (<= one sentence) or 'clean'\n\n"
        "Score: count captured out of 9.\n"
        "10 = all 9, 7 = 6-7, 5 = 4-5, 3 = 2-3, 1 = 0-1"
    ),
    max_turns=15,
    stage_limits={"spec_capture": 12},
    seed_fn=lambda project_dir: Path(project_dir / "prior_conversation.md").write_text(
        _MODERATION_CONVERSATION, encoding="utf-8"
    ),
)


SCENARIOS = [
    SPEC_GEN_NLP_TASK,
    SPEC_GEN_TOOL_CALLING,
    SPEC_GEN_SUMMARIZATION,
    SPEC_GEN_MODERATION,
]
