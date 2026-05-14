# Tool-calling pipeline for remote E2E testing.
#
# Designed to be non-trivial: 6 tools with multiple required parameters,
# overlapping domains, and user queries that require argument extraction
# from natural language (dates, times, formats, numeric ranges).
import json
import liquidrandom
from lqh.pipeline import (
    Pipeline, ChatMLMessage, Conversation, FunctionCall,
    GenerationError, ToolCall, ToolDef, step,
)

TOOLS = [
    ToolDef(
        name="search_flights",
        description="Search for available flights between two airports.",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Origin airport IATA code (e.g. SFO, JFK)"},
                "destination": {"type": "string", "description": "Destination airport IATA code"},
                "date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
                "passengers": {"type": "integer", "description": "Number of passengers"},
                "cabin_class": {
                    "type": "string",
                    "enum": ["economy", "premium_economy", "business", "first"],
                    "description": "Cabin class preference",
                },
            },
            "required": ["origin", "destination", "date", "passengers"],
        },
    ),
    ToolDef(
        name="book_hotel",
        description="Book a hotel room in a city for specific dates.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "check_in": {"type": "string", "description": "Check-in date in YYYY-MM-DD format"},
                "check_out": {"type": "string", "description": "Check-out date in YYYY-MM-DD format"},
                "guests": {"type": "integer", "description": "Number of guests"},
                "max_price_per_night": {"type": "number", "description": "Maximum price per night in USD"},
            },
            "required": ["city", "check_in", "check_out", "guests"],
        },
    ),
    ToolDef(
        name="convert_currency",
        description="Convert an amount from one currency to another.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code (e.g. USD, EUR)"},
                "to_currency": {"type": "string", "description": "Target currency code"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    ),
    ToolDef(
        name="schedule_meeting",
        description="Schedule a meeting with participants at a specific time.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Meeting title"},
                "date": {"type": "string", "description": "Meeting date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Start time in HH:MM format (24h)"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participant email addresses",
                },
            },
            "required": ["title", "date", "time", "duration_minutes", "participants"],
        },
    ),
    ToolDef(
        name="analyze_data",
        description="Run statistical analysis on a named dataset with specified metrics.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Name of the dataset to analyze"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["mean", "median", "std", "min", "max", "correlation", "regression"]},
                    "description": "Statistical metrics to compute",
                },
                "group_by": {"type": "string", "description": "Column name to group results by"},
                "filter_column": {"type": "string", "description": "Column to filter on"},
                "filter_value": {"type": "string", "description": "Value to filter by"},
            },
            "required": ["dataset_name", "metrics"],
        },
    ),
    ToolDef(
        name="send_notification",
        description="Send a notification message to a user or channel via a specific platform.",
        parameters={
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["email", "slack", "sms", "push"],
                    "description": "Notification platform",
                },
                "recipient": {"type": "string", "description": "Recipient address/ID (email, phone, Slack channel)"},
                "subject": {"type": "string", "description": "Message subject or title"},
                "body": {"type": "string", "description": "Message body content"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Message priority level",
                },
            },
            "required": ["platform", "recipient", "subject", "body"],
        },
    ),
]

# Scenarios that require careful argument extraction from natural language.
# Each maps a tool to templates with embedded parameters the model must extract.
SCENARIOS = {
    "search_flights": [
        "I need to fly from {origin} to {dest} on {date_natural} for {n} people, preferably {cabin}",
        "Can you find {cabin} class flights from {origin} to {dest}? It's for {n} of us on {date_natural}",
        "Looking for flights: {origin} -> {dest}, {date_natural}, {n} travelers",
    ],
    "book_hotel": [
        "I need a hotel in {city} from {checkin_natural} to {checkout_natural} for {n} guests, budget {price} per night",
        "Book me a room in {city}, checking in {checkin_natural} and out {checkout_natural}. {n} guests, under ${price}/night",
        "Find hotels in {city} for {n} people. Arriving {checkin_natural}, leaving {checkout_natural}. Max ${price}",
    ],
    "convert_currency": [
        "How much is {amount} {from_curr} in {to_curr}?",
        "Convert {amount} {from_curr} to {to_curr} for me",
        "I have {amount} {from_curr}, what's that in {to_curr}?",
    ],
    "schedule_meeting": [
        "Set up a {duration}min meeting called '{title}' on {date_natural} at {time} with {participants}",
        "Schedule '{title}' for {date_natural} {time}, {duration} minutes, invite {participants}",
        "I need a meeting: '{title}', {date_natural} at {time}, {duration}min, attendees: {participants}",
    ],
    "analyze_data": [
        "Run {metrics} on the {dataset} dataset, grouped by {group_by}",
        "Analyze {dataset}: compute {metrics}, filter where {filter_col} = {filter_val}",
        "Get {metrics} stats from {dataset}, break down by {group_by}",
    ],
    "send_notification": [
        "Send a {priority} priority {platform} to {recipient}: subject '{subject}', body: {body}",
        "Notify {recipient} via {platform} ({priority} priority) - '{subject}': {body}",
        "Message {recipient} on {platform}: '{subject}' - {body}. Mark as {priority}",
    ],
}


class ToolCallingTestPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.target_tool = liquidrandom.random.choice(TOOLS)
        self.persona = liquidrandom.persona()
        self.seed = f"tc-{self.target_tool.name}-{self.persona.name}"

        await self._gen_query(client)
        await self._gen_tool_interaction(client)
        await self._gen_response(client)

        return [
            ChatMLMessage(
                role="system",
                content=(
                    "You are a helpful assistant with access to tools. "
                    "When the user needs something that requires a tool, call the "
                    "appropriate tool with the correct arguments extracted from the "
                    "user's message. Pay careful attention to required parameter "
                    "formats (dates as YYYY-MM-DD, times as HH:MM, currency codes, "
                    "airport IATA codes, etc)."
                ),
                tools=TOOLS,
            ),
            ChatMLMessage(role="user", content=self.user_query),
            ChatMLMessage(
                role="assistant",
                content=self.prelude,
                tool_calls=[ToolCall(
                    id=f"call_{self.target_tool.name}_0",
                    function=FunctionCall(
                        name=self.target_tool.name,
                        arguments=self.tool_args_json,
                    ),
                )],
            ),
            ChatMLMessage(
                role="tool",
                content=self.tool_result,
                tool_call_id=f"call_{self.target_tool.name}_0",
                name=self.target_tool.name,
            ),
            ChatMLMessage(role="assistant", content=self.final_response),
        ]

    @step(retries=3)
    async def _gen_query(self, client):
        tool_name = self.target_tool.name
        templates = SCENARIOS[tool_name]
        template = liquidrandom.random.choice(templates)

        # Generate fill values
        rng = liquidrandom.random
        cities = ["Tokyo", "London", "Paris", "Berlin", "Sydney", "Toronto", "Dubai", "Singapore"]
        airports = ["SFO", "JFK", "LAX", "ORD", "LHR", "NRT", "CDG", "SIN", "DXB", "YYZ"]
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "SGD"]
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        datasets = ["q4_sales", "customer_churn", "website_traffic", "employee_survey", "product_reviews"]
        platforms = ["email", "slack", "sms", "push"]
        priorities = ["low", "normal", "high", "urgent"]

        day = rng.randint(1, 28)
        month = rng.choice(months)
        year = rng.choice([2025, 2026])
        day2 = min(day + rng.randint(2, 7), 28)

        fill = {
            "origin": rng.choice(airports),
            "dest": rng.choice(airports),
            "date_natural": f"{month} {day}, {year}",
            "checkin_natural": f"{month} {day}",
            "checkout_natural": f"{month} {day2}",
            "n": rng.randint(1, 6),
            "cabin": rng.choice(["economy", "business", "first"]),
            "city": rng.choice(cities),
            "price": rng.choice([100, 150, 200, 250, 300, 500]),
            "amount": round(rng.uniform(50, 10000), 2),
            "from_curr": rng.choice(currencies),
            "to_curr": rng.choice(currencies),
            "duration": rng.choice([15, 30, 45, 60, 90]),
            "title": rng.choice(["Q4 Planning", "Sprint Review", "1:1 Sync", "Budget Review", "Product Launch"]),
            "time": f"{rng.randint(8, 17):02d}:{rng.choice(['00', '15', '30', '45'])}",
            "participants": ", ".join(
                f"{liquidrandom.persona().name.split()[0].lower()}@company.com"
                for _ in range(rng.randint(2, 4))
            ),
            "metrics": ", ".join(rng.sample(["mean", "median", "std", "min", "max", "correlation"], k=rng.randint(2, 4))),
            "dataset": rng.choice(datasets),
            "group_by": rng.choice(["region", "department", "product", "quarter"]),
            "filter_col": rng.choice(["status", "category", "tier"]),
            "filter_val": rng.choice(["active", "premium", "enterprise"]),
            "platform": rng.choice(platforms),
            "recipient": f"{liquidrandom.persona().name.split()[0].lower()}@company.com",
            "subject": rng.choice(["Deployment Complete", "Action Required", "Weekly Update", "Incident Alert"]),
            "body": "",  # will be generated
            "priority": rng.choice(priorities),
        }
        self._fill = fill

        # Use LLM to make the query sound natural (not template-y)
        raw_query = template.format(**fill)
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    f"Rewrite this request as a natural, conversational message from a "
                    f"{self.persona.brief()}. Keep ALL the specific details (dates, numbers, "
                    f"names, codes) but make it sound like a real person talking. "
                    f"1-3 sentences, casual tone.\n\n"
                    f"Original: {raw_query}\n\n"
                    f"Rewritten (output ONLY the rewritten message):"
                ),
            }],
        )
        self.user_query = (resp.choices[0].message.content or "").strip().strip("'\"")
        if len(self.user_query) < 15:
            raise GenerationError("Query too short")

    @step(retries=3)
    async def _gen_tool_interaction(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[{
                "role": "user",
                "content": (
                    "Given this user message: " + repr(self.user_query) + "\n"
                    f"And this tool: {self.target_tool.name} with parameters: "
                    f"{json.dumps(self.target_tool.parameters)}\n\n"
                    "Generate JSON with:\n"
                    "1. \"arguments\": correct arguments matching the schema. "
                    "Convert natural language dates to YYYY-MM-DD, times to HH:MM, "
                    "use proper IATA codes, currency codes, etc.\n"
                    "2. \"result\": a realistic mock result the tool would return "
                    "(make it detailed with 2-3 items/options)\n"
                    "3. \"prelude\": a brief assistant message before calling the tool\n\n"
                    "Output ONLY valid JSON."
                ),
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        if "arguments" not in data or "result" not in data:
            raise GenerationError(f"Missing keys: {list(data.keys())}")
        self.tool_args_json = json.dumps(data["arguments"], ensure_ascii=False)
        result = data["result"]
        self.tool_result = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
        self.prelude = data.get("prelude", "Let me look that up for you.")

    @step(retries=3)
    async def _gen_response(self, client):
        resp = await client.chat.completions.create(
            model=f"random:small:{self.seed}",
            messages=[
                {"role": "system", "content": "Summarize tool results in a helpful, conversational way."},
                {"role": "user", "content": (
                    "User asked: " + repr(self.user_query) + "\n"
                    f"Tool {self.target_tool.name} returned: {self.tool_result}\n\n"
                    "Write a friendly 2-4 sentence response summarizing the results. "
                    "Include specific details from the tool output."
                )},
            ],
        )
        self.final_response = (resp.choices[0].message.content or "").strip()
        if len(self.final_response) < 20:
            raise GenerationError("Response too short")
