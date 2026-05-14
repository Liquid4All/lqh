"""Customer-feedback-bot data generation pipeline.

Produces 5-message ChatML conversations exercising one of 5 tools per
sample. The conversation shape matches what
``tests/remote/tool_calling_e2e_pipeline.py`` uses, so it flows
through ``lqh.infer`` and ``lqh.train`` unchanged. Tools are attached
to the *system* message — the engine writes them into the parquet's
``tools`` column.
"""

from __future__ import annotations

import json
import random

import liquidrandom

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    FunctionCall,
    GenerationError,
    Pipeline,
    ToolCall,
    ToolDef,
    safe_content,
    step,
)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


TOOLS = [
    ToolDef(
        name="lookup_order",
        description=(
            "Look up an order by ID and return its status, items, "
            "shipping, and tracking info."
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Alphanumeric order ID (e.g. 'ORD-A1B2C3').",
                },
            },
            "required": ["order_id"],
        },
    ),
    ToolDef(
        name="get_customer_history",
        description=(
            "Retrieve a customer's recent orders and support interactions "
            "within a time window."
        ),
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID (e.g. 'CUST-9341').",
                },
                "time_window": {
                    "type": "string",
                    "enum": ["7d", "30d", "90d", "1y", "all"],
                    "description": "How far back to pull.",
                },
            },
            "required": ["customer_id", "time_window"],
        },
    ),
    ToolDef(
        name="log_feedback",
        description=(
            "Record customer feedback for product/operations analytics. "
            "Use whenever the customer reports a bug, complaint, or "
            "suggestion that the company should track."
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "bug", "delivery", "quality", "billing",
                        "ux", "feature_request", "other",
                    ],
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "summary": {
                    "type": "string",
                    "description": "1-sentence summary of the feedback.",
                },
            },
            "required": ["category", "severity", "summary"],
        },
    ),
    ToolDef(
        name="send_acknowledgment",
        description=(
            "Send a canned acknowledgment email to a customer using a "
            "predefined template."
        ),
        parameters={
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "enum": [
                        "receipt_resend",
                        "refund_initiated",
                        "delay_apology",
                        "feedback_received",
                        "thanks_for_kind_words",
                    ],
                    "description": "ID of the canned template to send.",
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID receiving the acknowledgment.",
                },
            },
            "required": ["template_id", "customer_id"],
        },
    ),
    ToolDef(
        name="escalate_to_human",
        description=(
            "Hand off the conversation to a human support agent. Use only "
            "when the customer explicitly asks for a human, threatens "
            "legal action, mentions a safety issue, or has an issue "
            "outside the bot's scope."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short reason for the escalation.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["normal", "high", "critical"],
                },
            },
            "required": ["reason", "urgency"],
        },
    ),
]


# --------------------------------------------------------------------------
# Scenario templates per tool. Each entry is a query template that the LLM
# rewrites into a natural customer message, plus the deterministic "right
# answer" arguments and a sketch of the tool's mock result.
# --------------------------------------------------------------------------


SCENARIOS: dict[str, list[dict]] = {
    "lookup_order": [
        {
            "query": "Where is order {order_id}? It was supposed to arrive {expected}.",
            "args_factory": "_lookup_args",
            "result_factory": "_lookup_result",
        },
        {
            "query": "Status update on {order_id}, please.",
            "args_factory": "_lookup_args",
            "result_factory": "_lookup_result",
        },
        {
            "query": "I can't find my order {order_id} anywhere — has it shipped?",
            "args_factory": "_lookup_args",
            "result_factory": "_lookup_result",
        },
    ],
    "get_customer_history": [
        {
            "query": "Can you pull up everything I've bought in the last {window_human}? My account is {customer_id}.",
            "args_factory": "_history_args",
            "result_factory": "_history_result",
        },
        {
            "query": "I'm customer {customer_id}. What were my last few orders?",
            "args_factory": "_history_args",
            "result_factory": "_history_result",
        },
        {
            "query": "Need a record of my interactions with you over the past {window_human}. Customer ID {customer_id}.",
            "args_factory": "_history_args",
            "result_factory": "_history_result",
        },
    ],
    "log_feedback": [
        {
            "query": "{complaint_text}",
            "args_factory": "_feedback_args",
            "result_factory": "_feedback_result",
        },
    ],
    "send_acknowledgment": [
        {
            "query": "Could you resend my receipt for order {order_id}? Customer ID {customer_id}.",
            "args_factory": "_ack_args_receipt",
            "result_factory": "_ack_result",
        },
        {
            "query": "Hey, I never got an apology email for the late shipment on {order_id}. Can you send one? I'm {customer_id}.",
            "args_factory": "_ack_args_apology",
            "result_factory": "_ack_result",
        },
        {
            "query": "Thanks for the great service! Customer {customer_id}.",
            "args_factory": "_ack_args_thanks",
            "result_factory": "_ack_result",
        },
    ],
    "escalate_to_human": [
        {
            "query": "I want to speak to a human supervisor about {issue}. Now.",
            "args_factory": "_escalate_args",
            "result_factory": "_escalate_result",
        },
        {
            "query": "If this isn't fixed today I'm contacting my lawyer. Issue: {issue}.",
            "args_factory": "_escalate_args_legal",
            "result_factory": "_escalate_result",
        },
        {
            "query": "This product is a safety hazard — {issue}. I need someone real to handle this.",
            "args_factory": "_escalate_args_safety",
            "result_factory": "_escalate_result",
        },
    ],
}


# --------------------------------------------------------------------------
# Helpers (factory bodies)
# --------------------------------------------------------------------------


def _rand_order_id() -> str:
    suffix = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
    return f"ORD-{suffix}"


def _rand_customer_id() -> str:
    return f"CUST-{random.randint(1000, 99999)}"


COMPLAINT_TEMPLATES = [
    ("bug",            "low",      "Search results don't update when I clear the filter on the product page."),
    ("bug",            "medium",   "Checkout button is greyed out on mobile Safari."),
    ("bug",            "high",     "I was charged twice for the same order yesterday."),
    ("delivery",       "medium",   "My package was marked delivered but it's not at my door."),
    ("delivery",       "low",      "Shipping confirmation came in but no tracking number."),
    ("delivery",       "high",     "Order arrived two weeks late and the items were damaged."),
    ("quality",        "medium",   "The fabric on the shirt I bought feels noticeably thinner than advertised."),
    ("quality",        "low",      "Packaging was excessive — three boxes for one small item."),
    ("billing",        "high",     "I see a recurring charge on my card I never signed up for."),
    ("billing",        "medium",   "Promo code didn't apply at checkout but I was told it would."),
    ("ux",             "low",      "It took me 4 clicks to find where to update my shipping address."),
    ("feature_request","low",      "Would love to be able to pause my subscription instead of cancelling."),
    ("other",          "low",      "Just letting you know your support email goes to spam in Gmail."),
]


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class CustomerFeedbackBotV1(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        # Pick a target tool with a slight weight toward lookup_order
        # and log_feedback (the most common real-world cases).
        self.target_tool: ToolDef = random.choices(
            TOOLS,
            weights=[3, 2, 3, 2, 1],  # order, history, feedback, ack, escalate
        )[0]
        scenarios = SCENARIOS[self.target_tool.name]
        self.scenario = random.choice(scenarios)

        self.persona = liquidrandom.persona()
        self.seed = f"fb-{self.target_tool.name}-{self.persona.name}"

        # Build the deterministic args + result first; user query
        # references them so they line up with what the tool will see.
        factory_name = self.scenario["args_factory"]
        result_factory_name = self.scenario["result_factory"]
        self.tool_args, self.template_fill = getattr(self, factory_name)()
        self.tool_args_json = json.dumps(self.tool_args, ensure_ascii=False)
        self.tool_result = getattr(self, result_factory_name)(self.tool_args)

        await self._gen_user_query(client)
        await self._gen_prelude_and_response(client)

        return [
            ChatMLMessage(
                role="system",
                content=(
                    "You are a customer feedback bot. Read the customer's "
                    "message, pick exactly one tool, extract well-formed "
                    "arguments, and write a friendly 2-4 sentence reply "
                    "after the tool returns. Never invent details the "
                    "tool didn't return."
                ),
                tools=TOOLS,
            ),
            ChatMLMessage(role="user", content=self.user_query),
            ChatMLMessage(
                role="assistant",
                content=self.prelude,
                tool_calls=[
                    ToolCall(
                        id=f"call_{self.target_tool.name}_0",
                        function=FunctionCall(
                            name=self.target_tool.name,
                            arguments=self.tool_args_json,
                        ),
                    )
                ],
            ),
            ChatMLMessage(
                role="tool",
                content=self.tool_result,
                tool_call_id=f"call_{self.target_tool.name}_0",
                name=self.target_tool.name,
            ),
            ChatMLMessage(role="assistant", content=self.final_response),
        ]

    # ------------------------------------------------------------------
    # Argument / result factories
    # ------------------------------------------------------------------

    def _lookup_args(self) -> tuple[dict, dict]:
        order_id = _rand_order_id()
        return (
            {"order_id": order_id},
            {
                "order_id": order_id,
                "expected": random.choice([
                    "yesterday", "last Tuesday", "two days ago",
                    "earlier this week",
                ]),
            },
        )

    def _lookup_result(self, args: dict) -> str:
        outcomes = [
            {
                "status": "in_transit",
                "carrier": random.choice(["UPS", "FedEx", "DHL", "USPS"]),
                "tracking_number": "1Z" + "".join(random.choices("0123456789", k=10)),
                "estimated_delivery": random.choice(["tomorrow", "in 2 days", "Friday"]),
                "items": ["1× Wireless Headphones (black)"],
            },
            {
                "status": "delivered",
                "delivered_at": "2026-04-22T16:42:00Z",
                "items": ["2× Coffee Capsules pack"],
                "left_at": "front door",
            },
            {
                "status": "delayed",
                "carrier": "FedEx",
                "delay_reason": "weather",
                "new_estimated_delivery": "next Monday",
                "items": ["1× Office Chair (mesh)"],
            },
            {
                "status": "not_found",
                "message": "No order with that ID exists in our system.",
            },
        ]
        return json.dumps(random.choice(outcomes), ensure_ascii=False)

    def _history_args(self) -> tuple[dict, dict]:
        customer_id = _rand_customer_id()
        window = random.choice(["30d", "90d", "1y"])
        window_human = {"30d": "month", "90d": "three months", "1y": "year"}[window]
        return (
            {"customer_id": customer_id, "time_window": window},
            {"customer_id": customer_id, "window_human": window_human},
        )

    def _history_result(self, args: dict) -> str:
        n = random.randint(2, 5)
        orders = []
        for _ in range(n):
            orders.append({
                "order_id": _rand_order_id(),
                "date": f"2026-0{random.randint(1, 4)}-{random.randint(1, 28):02d}",
                "amount_usd": round(random.uniform(20, 400), 2),
                "status": random.choice(["delivered", "delivered", "returned", "refunded"]),
            })
        result = {
            "customer_id": args["customer_id"],
            "time_window": args["time_window"],
            "total_orders": n,
            "orders": orders,
            "open_support_tickets": random.choice([0, 0, 1, 2]),
        }
        return json.dumps(result, ensure_ascii=False)

    def _feedback_args(self) -> tuple[dict, dict]:
        category, severity, summary = random.choice(COMPLAINT_TEMPLATES)
        return (
            {"category": category, "severity": severity, "summary": summary},
            {"complaint_text": summary},
        )

    def _feedback_result(self, args: dict) -> str:
        return json.dumps(
            {
                "logged": True,
                "ticket_id": "FB-" + str(random.randint(10000, 99999)),
                "category": args["category"],
                "severity": args["severity"],
            },
            ensure_ascii=False,
        )

    def _ack_args_receipt(self) -> tuple[dict, dict]:
        order_id = _rand_order_id()
        customer_id = _rand_customer_id()
        return (
            {"template_id": "receipt_resend", "customer_id": customer_id},
            {"order_id": order_id, "customer_id": customer_id},
        )

    def _ack_args_apology(self) -> tuple[dict, dict]:
        order_id = _rand_order_id()
        customer_id = _rand_customer_id()
        return (
            {"template_id": "delay_apology", "customer_id": customer_id},
            {"order_id": order_id, "customer_id": customer_id},
        )

    def _ack_args_thanks(self) -> tuple[dict, dict]:
        customer_id = _rand_customer_id()
        return (
            {"template_id": "thanks_for_kind_words", "customer_id": customer_id},
            {"customer_id": customer_id},
        )

    def _ack_result(self, args: dict) -> str:
        return json.dumps(
            {
                "sent": True,
                "template_id": args["template_id"],
                "customer_id": args["customer_id"],
                "channel": "email",
            },
            ensure_ascii=False,
        )

    def _escalate_args(self) -> tuple[dict, dict]:
        issues = [
            "a refund that's been pending for three weeks",
            "an account lock that I can't resolve myself",
            "double-billing on my last invoice",
        ]
        issue = random.choice(issues)
        return (
            {"reason": issue, "urgency": "high"},
            {"issue": issue},
        )

    def _escalate_args_legal(self) -> tuple[dict, dict]:
        issues = [
            "a charge I never authorised",
            "personal data shown to the wrong account",
            "a defective product that injured my child",
        ]
        issue = random.choice(issues)
        return (
            {"reason": f"customer mentions legal action — {issue}", "urgency": "critical"},
            {"issue": issue},
        )

    def _escalate_args_safety(self) -> tuple[dict, dict]:
        issues = [
            "a battery in the device started smoking",
            "the food product had a sharp metal fragment in it",
            "the appliance shocked me when I plugged it in",
        ]
        issue = random.choice(issues)
        return (
            {"reason": f"safety issue — {issue}", "urgency": "critical"},
            {"issue": issue},
        )

    def _escalate_result(self, args: dict) -> str:
        return json.dumps(
            {
                "escalated": True,
                "case_id": "CASE-" + str(random.randint(10000, 99999)),
                "assigned_team": random.choice(["tier2", "trust_safety", "legal", "vip_care"]),
                "estimated_response_time_hours": random.choice([1, 2, 4, 24]),
                "urgency": args["urgency"],
            },
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # LLM steps
    # ------------------------------------------------------------------

    @step(retries=3)
    async def _gen_user_query(self, client):
        """Use the LLM to turn the templated query into a natural message."""
        raw = self.scenario["query"].format(**self.template_fill)
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Rewrite this customer-support request as a natural, "
                    f"realistic message from a {self.persona.brief()}. "
                    f"Keep ALL specific identifiers exactly (order IDs, "
                    f"customer IDs, time windows). Tone can vary: "
                    f"polite, frustrated, neutral, urgent — pick one. "
                    f"1-3 sentences. Output ONLY the rewritten message.\n\n"
                    f"Original: {raw}"
                ),
            }],
        )
        content = safe_content(resp).strip().strip("'\"")
        if len(content) < 10:
            raise GenerationError("User query too short")
        # Sanity: the rewriter sometimes mangles IDs. Verify they're present.
        for key in ("order_id", "customer_id"):
            tmpl_val = self.template_fill.get(key) or self.tool_args.get(key)
            if tmpl_val and tmpl_val in self.scenario["query"] and tmpl_val not in content:
                raise GenerationError(
                    f"Rewriter dropped {key}={tmpl_val!r} from query: {content[:120]}"
                )
        self.user_query = content

    @step(retries=3)
    async def _gen_prelude_and_response(self, client):
        """Generate the pre-tool prelude + the post-tool customer reply."""
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Customer message: {self.user_query!r}\n"
                    f"Tool called: {self.target_tool.name}\n"
                    f"Tool result: {self.tool_result}\n\n"
                    "Output JSON with two keys:\n"
                    "  \"prelude\": one short sentence the bot says BEFORE "
                    "calling the tool (e.g. \"Let me check that for you.\").\n"
                    "  \"final\": the bot's 2-4 sentence customer-facing "
                    "reply AFTER the tool returns. Friendly, professional, "
                    "grounded in the tool result. No preamble. No promises "
                    "the tool didn't authorise.\n"
                    "Output ONLY valid JSON."
                ),
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(safe_content(resp) or "{}")
        prelude = (data.get("prelude") or "").strip()
        final = (data.get("final") or "").strip()
        if len(prelude) < 5 or len(prelude) > 200:
            raise GenerationError(f"Prelude length {len(prelude)} out of range")
        if len(final) < 30 or len(final) > 600:
            raise GenerationError(f"Final length {len(final)} out of range")
        self.prelude = prelude
        self.final_response = final
