"""Quick test: verify the jinja chat template renders correctly."""

from jinja2 import Environment

TEMPLATE_PATH = "tool_chat_template_qwen2_5_coder.jinja"

with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_str = f.read()

env = Environment()
template = env.from_string(template_str)

# Test 1: Basic tool call scenario
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Seoul?"},
]

result = template.render(
    messages=messages,
    tools=tools,
    add_generation_prompt=True,
    bos_token="",
)

print("=== Test 1: User message with tools ===")
print(result)
print()

# Test 2: Multi-turn with tool call and response
messages_multi = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "What's the weather in Seoul?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Seoul"}',
                }
            }
        ],
    },
    {"role": "tool", "content": '{"temperature": 15, "condition": "cloudy"}'},
    {"role": "assistant", "content": "The weather in Seoul is 15°C and cloudy."},
    {"role": "user", "content": "How about Tokyo and Paris?"},
]

result2 = template.render(
    messages=messages_multi,
    tools=tools,
    add_generation_prompt=True,
    bos_token="",
)

print("=== Test 2: Multi-turn with tool calls ===")
print(result2)
print()

# Test 3: No tools (should not add few-shot examples)
messages_no_tools = [
    {"role": "user", "content": "Hello!"},
]

result3 = template.render(
    messages=messages_no_tools,
    tools=[],
    add_generation_prompt=True,
    bos_token="",
)

print("=== Test 3: No tools ===")
print(result3)
