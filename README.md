# Qwen2.5-Coder `<tools>` Tag Parser for vLLM

A vLLM tool parser for Qwen2.5-Coder models that use `<tools>` tag format for tool calling.

## Problem

[vLLM's documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html) suggests using `--tool-call-parser hermes` for Qwen2.5 models, but **Qwen2.5-Coder models do not use the hermes `<tool_call>` format**. This means tool calling fails silently — the model outputs tool calls in a different format that vLLM cannot parse.

We found that without any format instruction, Qwen2.5-Coder outputs tool calls as ` ```json ``` ` code blocks. vLLM cannot parse these as native tool calls — they arrive as plain `content` text. While a server-side text fallback parser could extract tool calls from code blocks, this approach is inherently fragile:
- No clear boundary between tool call JSON and other code blocks in the response
- Ambiguous when the model outputs JSON for explanation vs. invocation
- Parallel tool calls in code blocks lack a consistent delimiter

We tested several prompting strategies and found that hermes-style `<tool_call>` prompting is ignored by the model (60% code blocks + 40% plain JSON), but `<tools>` tag few-shot examples achieve 100% format compliance. (See [Background: Why not hermes?](#background-why-not-hermes) for details.)

The solution is to prompt the model to use `<tools>` tags, which provide unambiguous start/end markers that vLLM can parse deterministically — just like hermes `<tool_call>` tags. The provided chat template automates this by injecting few-shot `<tools>` examples into the system prompt when tools are present in the API request.

Related issues: [vLLM #10952](https://github.com/vllm-project/vllm/issues/10952), [vLLM #29192](https://github.com/vllm-project/vllm/issues/29192)

## Usage

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin qwen2_5_coder_tool_parser.py \
    --tool-call-parser qwen2_5_coder \
    --chat-template tool_chat_template_qwen2_5_coder.jinja
```

## Supported Formats

The parser handles all observed output patterns from Qwen2.5-Coder:

### Single tool call
```
<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tools>
```

### Parallel tool calls (repeated tags)
```
<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tools>
<tools>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tools>
```

### Parallel tool calls (JSON array)
```
<tools>[
  {"name": "get_weather", "arguments": {"city": "Seoul"}},
  {"name": "get_weather", "arguments": {"city": "Tokyo"}}
]</tools>
```

### JSONL (newline-separated)
```
<tools>
{"name": "get_weather", "arguments": {"city": "Seoul"}}
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tools>
```

### Unclosed tag (streaming edge case)
```
<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}
```

## Testing

### Unit tests (no server required)

Validates parser logic in isolation with 80 test cases.

```bash
pytest tests/test_parser_unit.py -v
```

Covers: single calls, parallel calls, JSON arrays, JSONL, comma-separated, unclosed tags, empty/invalid input, whitespace variations, double-escaped JSON, mixed content.

### Output format analysis (requires vLLM server without `--chat-template`)

Discovers how the model outputs tool calls under different system prompt strategies. The server must NOT use `--chat-template`, as it would inject `<tools>` few-shot examples and contaminate the format comparison.

```bash
# Server without --chat-template
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin qwen2_5_coder_tool_parser.py \
    --tool-call-parser qwen2_5_coder

python tests/test_output_format.py --model Qwen/Qwen2.5-Coder-7B-Instruct
```

### Integration tests (requires vLLM server with parser + chat template)

End-to-end validation that vLLM + parser + chat template correctly returns `tool_calls`.

```bash
python tests/test_parser_vllm_integration.py --model Qwen/Qwen2.5-Coder-7B-Instruct
```

Tests 50 cases: single tool calls (10), parallel calls (10), complex arguments (10), edge cases (10), no-tool queries (10).

## Tested Models

| Model | End-to-End | Notes |
|-------|:----------:|-------|
| Qwen/Qwen2.5-Coder-7B-Instruct | ✅ 50/50 | |
| Qwen/Qwen2.5-Coder-14B-Instruct | 48/49* | 1 JSON malformation (model limitation) |
| Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | ✅ 50/50 | |

Parser success rate is 100% across all models — all `<tools>` formatted output was parsed correctly. The one failure is a model-level JSON generation issue (JSON-in-JSON escaping: model outputs `}}}` instead of `}}`), not a parser issue.

\* 1 additional case excluded: model answered directly without calling tools (model behavior).

**Test environment:** vLLM 0.14.0, `temperature=0`, `max_tokens=1024`, `--chat-template`

**Raw results:** [`results/`](results/)

**Not applicable:**
- Qwen2.5 (non-Coder) — uses hermes `<tool_call>` natively, works with `--tool-call-parser hermes`
- Qwen3 (non-Coder) — uses hermes `<tool_call>` natively, works with `--tool-call-parser hermes`
- Qwen3-Coder (30B-A3B) — uses XML format, works with `--tool-call-parser qwen3_coder`

## How It Works

1. Chat template injects tool definitions and few-shot `<tools>` examples into system prompt
2. Model outputs: `<tools>{"name": "...", "arguments": {...}}</tools>`
3. Parser extracts JSON from `<tools>` tags via regex
4. Parsed tool calls are returned as OpenAI-compatible `tool_calls` objects
5. Text outside `<tools>` tags is returned as regular `content`

Both streaming (`extract_tool_calls_streaming`) and non-streaming (`extract_tool_calls`) modes are supported.

## Project Structure

```
├── qwen2_5_coder_tool_parser.py    # vLLM parser plugin (register as "qwen2_5_coder")
├── tool_chat_template_qwen2_5_coder.jinja  # Chat template (auto-injects few-shot examples)
├── README.md
├── fixtures/
│   ├── test_cases.yaml             # 50 test cases across 5 categories
│   └── test_tools.yaml             # 10 tool definitions for testing
├── tests/
│   ├── test_parser_unit.py         # 80 unit tests (no server required)
│   ├── test_parser_vllm_integration.py  # Integration tests (requires vLLM)
│   └── test_output_format.py       # Output format analysis
└── results/
    ├── output_format/              # Format analysis results by model
    ├── integration_template/       # Chat template mode (final solution)
    ├── integration_minimal/        # Few-shot only in system prompt (no format instructions)
    ├── integration_explicit/       # Explicit format instructions in system prompt
    └── integration_verbose/        # Long system prompt + few-shot in system prompt
```

## License

Apache 2.0 (same as vLLM)

---

<a id="background-why-not-hermes"></a>
<details>
<summary><strong>Background: Why not hermes?</strong></summary>

### Format comparison

We tested Qwen2.5-Coder-7B-Instruct with three system prompt strategies to determine its native tool calling format:

| Strategy | Description | Result |
|----------|-------------|--------|
| **No instruction** | No format guidance in system prompt | 100% ` ```json ``` ` code blocks |
| **Hermes induction** | `<tool_call>` tag examples in system prompt | 60% code blocks + 40% plain JSON (ignores hermes) |
| **`<tools>` induction** | `<tools>` tag examples in system prompt | **100% success** (vLLM native parsing) |

The model completely ignores hermes-style `<tool_call>` prompting but perfectly follows `<tools>` tag format when instructed with a few-shot example. ([raw results](results/output_format/))

### Qwen2.5 (non-Coder)

Qwen2.5-7B-Instruct (non-Coder) uses hermes `<tool_call>` format natively regardless of system prompt. It works correctly with vLLM's built-in `--tool-call-parser hermes`. This parser is specifically for the Coder variant which ignores hermes format.

</details>

<details>
<summary><strong>Long System Prompt Robustness</strong></summary>

Since the parser relies on few-shot examples in the system prompt to induce `<tools>` format, we tested whether long system prompts degrade the induction effectiveness. These tests were run without `--chat-template`, using manual few-shot injection via `--prompt-mode`:

```bash
# Server without --chat-template
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin qwen2_5_coder_tool_parser.py \
    --tool-call-parser qwen2_5_coder

# Test with manual few-shot modes (minimal, explicit, or verbose)
python tests/test_parser_vllm_integration.py --model Qwen/Qwen2.5-Coder-7B-Instruct --prompt-mode minimal
python tests/test_parser_vllm_integration.py --model Qwen/Qwen2.5-Coder-7B-Instruct --prompt-mode explicit
python tests/test_parser_vllm_integration.py --model Qwen/Qwen2.5-Coder-7B-Instruct --prompt-mode verbose
```

### Explicit mode issue ([raw results](results/integration_explicit/))

With `explicit` mode (system prompt with detailed `<tools>` format instructions), 14B deterministically fails 1 case (49/50). The failure is the "JSON content in argument" test case (`Write to output.json the content: {"name": "test", "value": 123}`) — the model generates `}}}` instead of `}}`. Switching to `minimal` (few-shot only) resolves this to 50/50, indicating that longer system prompts can degrade JSON generation quality.

### Verbose mode results ([raw results](results/integration_verbose/))

Tested with a ~90-line system prompt (software development assistant with code style guides, security rules, testing standards) combined with few-shot examples.

| Model | Parser Success | Tool Not Called | JSON Malformation |
|-------|:--------------:|:---------------:|:-----------------:|
| Qwen2.5-Coder-7B-Instruct | 35/36 (97.2%) | 4 | 1 |
| Qwen2.5-Coder-14B-Instruct | 39/40 (97.5%) | 0 | 1 |
| Qwen2.5-Coder-32B-Instruct-AWQ | 39/40 (97.5%) | 0 | 1 |

- **Format induction**: 14B and 32B use `<tools>` tags 100% of the time even with a verbose system prompt. Few-shot examples are not buried by long instructions.
- **JSON malformation**: The same "JSON-in-argument" case fails identically across all 3 models. This is a model-level JSON-in-JSON escaping limitation, not a parser issue.
- **Tool not called (7B only)**: 7B does not call tools in 4/40 cases with verbose prompts, answering directly instead. This is model behavior degradation with long system prompts on smaller parameter models, not a parser issue.

</details>
