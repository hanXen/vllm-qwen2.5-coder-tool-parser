"""
vLLM qwen2_5_coder parser integration tests

Tests that the qwen2_5_coder parser correctly parses tool call output from vLLM.
Focuses on parser quality: valid JSON arguments, valid function names,
and correct argument values per selected tool.

Test categories:
1. single - Single tool call (basic parsing)
2. parallel - Multiple tool calls in one response (multi-object parsing)
3. complex_args - Complex arguments (newlines, quotes, unicode, special chars)
4. edge_cases - Edge cases for parser robustness (typos, minimal input, etc.)
5. no_tool - Should NOT trigger tool calling (side-effect detection)

Usage:
    uv run python test_vllm_parser.py --api-base http://localhost:8000/v1
    uv run python test_vllm_parser.py --repeat 3  # Repeat each query 3 times
    uv run python test_vllm_parser.py --category single  # Run only single tests
    uv run python test_vllm_parser.py --prompt-mode minimal  # Few-shot only prompt
"""

# pylint: disable=line-too-long
# pylint: disable=missing-class-docstring,missing-function-docstring

import argparse
import asyncio
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import AsyncOpenAI

# Windows terminal UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestCase:
    """Test case definition"""
    category: str  # single, parallel, complex_args, edge_cases, no_tool
    query: str
    # tool_name -> {arg: pattern}
    arg_checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str = ""


@dataclass
class TestResult:
    """Test result"""
    test_case: TestCase
    success: bool
    tool_calls_count: int
    tool_names: list[str]
    tool_args: list[dict[str, Any]]  # Actual parsed arguments
    content: Optional[str]
    error: Optional[str] = None
    args_validation: Optional[list[dict]] = None  # Per-tool-call validation details
    parse_errors: list[str] = field(default_factory=list)


# =============================================================================
# Argument Validation
# =============================================================================

def validate_arg_pattern(actual_value: Any, pattern: Any) -> tuple[bool, str]:
    """
    Validate a single argument value against a pattern.

    Pattern types:
    - Exact value: "Seoul" -> must equal "Seoul"
    - Contains: {"contains": "Seoul"} -> must contain "Seoul"
    - Regex: {"regex": "^Seoul$"} -> must match regex
    - Required: {"required": true} -> just check existence

    Returns: (success, reason)
    """
    if actual_value is None:
        return False, "argument is missing"

    actual_str = str(actual_value)

    # Pattern is a dict with validation type
    if isinstance(pattern, dict):
        if "contains" in pattern:
            substr = pattern["contains"]
            if substr in actual_str:
                return True, f"contains '{substr}'"
            return False, f"does not contain '{substr}' (got: '{actual_str[:50]}')"

        if "regex" in pattern:
            regex = pattern["regex"]
            if re.search(regex, actual_str):
                return True, f"matches regex '{regex}'"
            return False, f"does not match regex '{regex}' (got: '{actual_str[:50]}')"

        if "required" in pattern and pattern["required"]:
            return True, "exists (required)"

        return False, f"unknown pattern type: {pattern}"

    # Pattern is a literal value - exact match
    if actual_str == str(pattern):
        return True, f"exact match '{pattern}'"
    return False, f"expected '{pattern}', got '{actual_str[:50]}'"


def validate_tool_args(
    actual_args: dict[str, Any],
    expected_pattern: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """
    Validate all arguments in a tool call against expected patterns.

    Returns: (all_valid, details)
    """
    details = {}
    all_valid = True

    for arg_name, pattern in expected_pattern.items():
        actual_value = actual_args.get(arg_name)
        valid, reason = validate_arg_pattern(actual_value, pattern)
        details[arg_name] = {"valid": valid, "reason": reason, "actual": actual_value}
        if not valid:
            all_valid = False

    return all_valid, details


def validate_tool_calls_against_checks(
    tool_calls: list[dict[str, Any]],
    arg_checks: dict[str, dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Validate each tool call's arguments against arg_checks.

    For each tool call:
    - If tool name has an entry in arg_checks, validate args against its patterns
    - If tool name has no entry, only JSON validity matters (already checked)

    Returns: (all_valid, validation_results)
    """
    if not arg_checks:
        return True, []

    validation_results = []
    all_valid = True

    for i, tc in enumerate(tool_calls):
        tool_name = tc["name"]
        pattern = arg_checks.get(tool_name)

        if pattern is None:
            # No arg_checks for this tool - skip (JSON validity already checked)
            validation_results.append({
                "index": i,
                "tool": tool_name,
                "valid": True,
                "details": {},
                "note": "no arg_checks defined for this tool",
            })
            continue

        # Parse arguments
        try:
            args = (json.loads(tc["arguments"])
                    if isinstance(tc["arguments"], str)
                    else tc["arguments"])
        except json.JSONDecodeError:
            validation_results.append({
                "index": i,
                "tool": tool_name,
                "valid": False,
                "details": {},
                "error": "JSON parse error in arguments",
            })
            all_valid = False
            continue

        # Validate against pattern
        valid, details = validate_tool_args(args, pattern)
        validation_results.append({
            "index": i,
            "tool": tool_name,
            "valid": valid,
            "details": details,
            "actual_args": args,
        })
        if not valid:
            all_valid = False

    return all_valid, validation_results


# =============================================================================
# YAML Loaders
# =============================================================================

def load_tools(yaml_path: Path) -> list[dict[str, Any]]:
    """Load tool definitions from YAML"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tools = []
    for tool in data.get("tools", []):
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        })
    return tools


def load_test_cases(yaml_path: Path) -> list[TestCase]:
    """Load test cases from YAML"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = []
    for category in ["single", "parallel", "complex_args", "edge_cases", "no_tool"]:
        for case in data.get(category, []):
            test_cases.append(TestCase(
                category=category,
                query=case["query"],
                arg_checks=case.get("arg_checks", {}),
                description=case.get("description", ""),
            ))
    return test_cases


# =============================================================================
# System Prompts
# =============================================================================

# Explicit: detailed <tools> tag usage instructions
SYSTEM_PROMPT_EXPLICIT = """You are a helpful assistant with access to various tools.
When you need to use a tool, you MUST output the tool call wrapped in <tools> tags like this:

<tools>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tools>

For multiple tool calls, you can use multiple <tools> tags:
<tools>
{"name": "tool1", "arguments": {...}}
</tools>
<tools>
{"name": "tool2", "arguments": {...}}
</tools>

Or use an array:
<tools>
[{"name": "tool1", "arguments": {...}}, {"name": "tool2", "arguments": {...}}]
</tools>

Available tools will be provided. Use them when appropriate."""

# Minimal: minimal instructions only
SYSTEM_PROMPT_MINIMAL = """You are a helpful assistant with access to various tools."""

# Few-shot examples for minimal prompt
FEWSHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "What's the weather in Tokyo?"
    },
    {
        "role": "assistant",
        "content": '<tools>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tools>'
    },
    {
        "role": "user",
        "content": "Thanks! How about in Paris and Berlin?"
    },
    {
        "role": "assistant",
        "content": (
            '<tools>\n{"name": "get_weather", '
            '"arguments": {"city": "Paris"}}\n</tools>\n'
            '<tools>\n{"name": "get_weather", '
            '"arguments": {"city": "Berlin"}}\n</tools>'
        ),
    },
]

# Verbose: realistic long system prompt + few-shot
# Purpose: verify few-shot <tools> induction works with long system prompts
SYSTEM_PROMPT_VERBOSE = """You are an expert software development assistant specializing in full-stack web development, system design, and code review. You have deep expertise in Python, TypeScript, Go, Rust, and their respective ecosystems.

## Core Responsibilities

1. **Code Analysis**: When asked to review code, provide detailed feedback on:
   - Code correctness and potential bugs
   - Performance implications and optimization opportunities
   - Security vulnerabilities (OWASP Top 10, injection attacks, auth issues)
   - Design patterns and architectural concerns
   - Test coverage gaps and edge cases

2. **Implementation**: When asked to implement features:
   - Follow SOLID principles and clean code practices
   - Write idiomatic code for the target language
   - Include appropriate error handling and logging
   - Consider backwards compatibility
   - Prefer composition over inheritance

3. **System Design**: When discussing architecture:
   - Consider scalability requirements (horizontal vs vertical scaling)
   - Evaluate trade-offs between consistency and availability (CAP theorem)
   - Recommend appropriate data stores for the use case
   - Design for observability (metrics, tracing, logging)
   - Plan for failure modes and graceful degradation

## Response Guidelines

- Always explain your reasoning before providing code
- Use code blocks with appropriate language tags for syntax highlighting
- When suggesting changes to existing code, show the diff or highlight modified sections
- If a question is ambiguous, ask for clarification before proceeding
- Provide references to documentation or RFCs when relevant
- Consider the user's experience level and adjust explanations accordingly

## Code Style Preferences

### Python
- Use type hints for all function signatures
- Follow PEP 8 and PEP 257 for formatting and docstrings
- Prefer pathlib over os.path for file operations
- Use dataclasses or Pydantic for structured data
- Async/await for I/O-bound operations

### TypeScript
- Strict mode enabled, no implicit any
- Prefer interfaces over type aliases for object shapes
- Use discriminated unions for state management
- Functional components with hooks for React
- Zod for runtime validation

### Go
- Follow Effective Go guidelines
- Use context.Context for cancellation and timeouts
- Table-driven tests with subtests
- Structured logging with slog
- Errors as values, wrap with fmt.Errorf

### Rust
- Prefer Result over panic for error handling
- Use clippy lints at pedantic level
- Lifetime annotations only when necessary
- Derive macros for common traits
- Document unsafe blocks with safety invariants

## Testing Standards

- Unit tests for pure business logic
- Integration tests for external dependencies
- Property-based tests for complex algorithms
- Benchmark tests for performance-critical paths
- Minimum 80% code coverage for new code

## Security Considerations

- Never log sensitive data (passwords, tokens, PII)
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Implement rate limiting for public APIs
- Follow principle of least privilege for permissions
- Use constant-time comparison for secrets

## Communication Style

- Be concise but thorough
- Use bullet points for lists of items
- Include examples when explaining concepts
- Acknowledge limitations or uncertainty
- Suggest alternatives when the requested approach has issues

You have access to various tools that can help you assist the user. Use them when the user's request requires real-time data, calculations, file operations, or other actions that cannot be answered from knowledge alone."""

SYSTEM_PROMPTS = {
    "explicit": SYSTEM_PROMPT_EXPLICIT,
    "minimal": SYSTEM_PROMPT_MINIMAL,
    "verbose": SYSTEM_PROMPT_VERBOSE,
    "template": None,  # No system prompt — chat template handles everything
}


# =============================================================================
# Test Runner
# =============================================================================

async def run_single_test(
    client: AsyncOpenAI,
    model: str,
    test_case: TestCase,
    tools: list[dict[str, Any]],
    prompt_mode: str = "explicit",
) -> TestResult:
    """Run a single test case"""
    try:
        # Build messages based on prompt mode
        system_prompt = SYSTEM_PROMPTS.get(prompt_mode, SYSTEM_PROMPT_EXPLICIT)
        messages = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        # Add few-shot examples for minimal and verbose modes
        if prompt_mode in ("minimal", "verbose"):
            messages.extend(FEWSHOT_EXAMPLES)

        messages.append({"role": "user", "content": test_case.query})

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0.0,
                max_tokens=1024,
            ),
            timeout=60.0,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        tool_names = [tc.function.name for tc in tool_calls]

        # Parse arguments and check parsing success
        tool_args = []
        tool_calls_raw = []
        parse_errors = []
        for tc in tool_calls:
            try:
                args = (json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments)
                if not isinstance(args, dict):
                    parse_errors.append(
                        f"arguments is not a dict: {type(args)}")
                    args = {"_raw": tc.function.arguments}
            except json.JSONDecodeError as e:
                parse_errors.append(f"JSON parse error: {e}")
                args = {"_raw": tc.function.arguments}
            tool_args.append(args)
            tool_calls_raw.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })

        # Success criteria
        args_validation = None

        if test_case.category == "no_tool":
            # no_tool: model shouldn't call tools, but if it does, check parsing
            if len(tool_calls) == 0:
                success = True
            else:
                # Model called a tool unexpectedly - still check parsing
                success = len(parse_errors) == 0 and all(
                    tc.function.name and isinstance(tc.function.name, str)
                    for tc in tool_calls
                )
        else:
            # Tool-call categories: parser must produce valid tool_calls
            if len(tool_calls) == 0:
                success = False
            else:
                # Base check: valid function names + valid JSON arguments
                success = (
                    len(parse_errors) == 0
                    and all(
                        tc.function.name
                        and isinstance(tc.function.name, str)
                        for tc in tool_calls
                    )
                )

            # Validate argument values per selected tool
            if success and test_case.arg_checks:
                args_valid, args_validation = (
                    validate_tool_calls_against_checks(
                        tool_calls_raw, test_case.arg_checks
                    )
                )
                success = success and args_valid

        return TestResult(
            test_case=test_case,
            success=success,
            tool_calls_count=len(tool_calls),
            tool_names=tool_names,
            tool_args=tool_args,
            content=message.content,
            args_validation=args_validation,
            parse_errors=parse_errors,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        return TestResult(
            test_case=test_case,
            success=False,
            tool_calls_count=0,
            tool_names=[],
            tool_args=[],
            content=None,
            error=str(e),
        )


async def run_tests(
    client: AsyncOpenAI,
    model: str,
    tests: list[TestCase],
    tools: list[dict[str, Any]],
    repeat: int = 1,
    prompt_mode: str = "explicit",
) -> list[TestResult]:
    """Run all test cases"""
    results = []

    for test_case in tests:
        for i in range(repeat):
            result = await run_single_test(
                client, model, test_case, tools, prompt_mode)
            results.append(result)

            # Print result
            status = "✅" if result.success else "❌"
            repeat_str = f" [{i+1}/{repeat}]" if repeat > 1 else ""
            print(f"{status} [{test_case.category}]{repeat_str} "
                  f"{test_case.query[:50]}...")

            if not result.success:
                if result.error:
                    print(f"   Error: {result.error}")
                else:
                    print(f"   Tools: {result.tool_names}")
                    for pe in result.parse_errors:
                        print(f"   Parse error: {pe}")
                    if result.tool_args:
                        args_str = json.dumps(
                            result.tool_args, ensure_ascii=False)[:150]
                        print(f"   Args: {args_str}")
                    if result.args_validation:
                        for v in result.args_validation:
                            if not v.get("valid"):
                                print(f"   Arg validation: "
                                      f"{v.get('tool')} - "
                                      f"{v.get('error', v.get('details'))}")
                    if result.content:
                        print(f"   Content: {result.content[:100]}...")

    return results


def print_summary(results: list[TestResult]):
    """Print result summary"""
    print(f"\n{'='*60}")
    print("Result Summary")
    print(f"{'='*60}")

    # Aggregate by category
    categories = {}
    for r in results:
        cat = r.test_case.category
        if cat not in categories:
            categories[cat] = {"success": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.success:
            categories[cat]["success"] += 1

    # Print per-category results
    for cat, counts in sorted(categories.items()):
        pct = counts["success"] / counts["total"] * 100
        print(f"  {cat}: {counts['success']}/{counts['total']}"
              f" ({pct:.1f}%)")

    # Total score
    total_success = sum(c["success"] for c in categories.values())
    total_count = sum(c["total"] for c in categories.values())

    if total_count > 0:
        print(f"\n  Total: {total_success}/{total_count}"
              f" ({total_success/total_count*100:.1f}%)")


async def main():
    parser = argparse.ArgumentParser(
        description="vLLM qwen2_5_coder parser integration tests")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="LLM API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat count for each test")
    parser.add_argument(
        "--category", default=None,
        help="Test category (single, parallel, complex_args, "
             "edge_cases, no_tool)")
    parser.add_argument(
        "--prompt-mode", default="template",
        choices=["template", "minimal", "explicit", "verbose"],
        help="Prompt mode: template (chat template only, default), "
             "minimal (few-shot messages), "
             "explicit (detailed format instructions), "
             "or verbose (long system prompt + few-shot)")
    args = parser.parse_args()

    api_base = args.api_base
    model = args.model

    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"Repeat: {args.repeat}")
    print(f"Prompt Mode: {args.prompt_mode}")
    print()

    client = AsyncOpenAI(base_url=api_base, api_key="dummy")

    # Load tools and test cases from YAML
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    tools = load_tools(fixtures_dir / "test_tools.yaml")
    all_tests = load_test_cases(fixtures_dir / "test_cases.yaml")

    print(f"Loaded {len(tools)} tools")
    print(f"Loaded {len(all_tests)} test cases")

    # Filter by category if specified
    if args.category:
        tests = [t for t in all_tests
                 if t.category == args.category]
    else:
        tests = all_tests

    print(f"Running {len(tests)} tests")
    print(f"{'='*60}")

    # Run tests
    results = await run_tests(
        client, model, tests, tools, args.repeat, args.prompt_mode)

    # Print summary
    print_summary(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results"
    output_file = output_dir / f"vllm_parser_test_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": timestamp,
        "model": model,
        "repeat": args.repeat,
        "prompt_mode": args.prompt_mode,
        "tools_count": len(tools),
        "tests_count": len(tests),
        "results": [
            {
                "category": r.test_case.category,
                "query": r.test_case.query,
                "description": r.test_case.description,
                "success": r.success,
                "tool_calls_count": r.tool_calls_count,
                "tool_names": r.tool_names,
                "tool_args": r.tool_args,
                "args_validation": r.args_validation,
                "parse_errors": r.parse_errors,
                "content": r.content,
                "error": r.error,
            }
            for r in results
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
