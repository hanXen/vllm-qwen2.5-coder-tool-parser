# Test Results

Raw JSON outputs from all tests. See [main README](../README.md) for methodology.

## Output Format Analysis (`output_format/`)

How models output tool calls under different system prompt strategies.

### Qwen2.5-Coder (target models — need this parser)

| Model | No instruction | Hermes `<tool_call>` | `<tools>` tag |
|-------|----------------|----------------------|---------------|
| [7B](output_format/qwen2.5-coder-7b.json) | 100% code blocks | 60% code blocks, 40% plain JSON | **100% `<tools>`** |
| [14B](output_format/qwen2.5-coder-14b.json) | 90% plain JSON, 10% code blocks | 100% plain JSON | **100% `<tools>`** |
| [32B-AWQ](output_format/qwen2.5-coder-32b-awq.json) | 100% plain JSON | 100% plain JSON | **100% `<tools>`** |

### Other models (use built-in parsers)

| Model | No instruction | Hermes `<tool_call>` | `<tools>` tag | Parser |
|-------|----------------|----------------------|---------------|--------|
| [Qwen3-8B](output_format/qwen3-8b.json) | 100% hermes | 100% hermes | 100% hermes | `hermes` |
| [Qwen3-Coder-30B-A3B-AWQ](output_format/qwen3-coder-30b-a3b-awq.json) | 100% XML | 80% XML, 20% `<tool_call>` | 40% XML, 60% `<tools>` | `qwen3_coder` |

Qwen3-8B always uses hermes format regardless of prompting. Qwen3-Coder partially responds to prompting but its native XML format works without any instruction.

## Integration Tests

### With chat template (`integration_template/`) — Main results

| Model | Result |
|-------|:------:|
| [7B](integration_template/qwen2.5-coder-7b.json) | 50/50 |
| [14B](integration_template/qwen2.5-coder-14b.json) | 48/49* |
| [32B-AWQ](integration_template/qwen2.5-coder-32b-awq.json) | 50/50 |

\* 1 JSON malformation (model limitation), 1 additional case excluded where model answered directly

### Without chat template (manual few-shot)

Test modes with different system prompt lengths:
- **minimal**: Few-shot examples only
- **explicit**: Few-shot + format instructions
- **verbose**: Few-shot + ~90 line system prompt (code style guides, security rules, etc.)

| Model | minimal | explicit | verbose |
|-------|:-------:|:--------:|:-------:|
| 7B | [50/50](integration_minimal/qwen2.5-coder-7b.json) | [50/50](integration_explicit/qwen2.5-coder-7b.json) | [45/46*](integration_verbose/qwen2.5-coder-7b.json) |
| 14B | [50/50](integration_minimal/qwen2.5-coder-14b.json) | [49/50](integration_explicit/qwen2.5-coder-14b.json) | [49/50](integration_verbose/qwen2.5-coder-14b.json) |
| 32B-AWQ | [50/50](integration_minimal/qwen2.5-coder-32b-awq.json) | [50/50](integration_explicit/qwen2.5-coder-32b-awq.json) | [49/50](integration_verbose/qwen2.5-coder-32b-awq.json) |

\* 4 additional cases excluded where model answered directly

All non-50/50 results have 1 JSON malformation (same "JSON-in-argument" test case, model limitation)
