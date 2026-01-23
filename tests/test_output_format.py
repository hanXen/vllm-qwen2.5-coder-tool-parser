"""
Output format analysis for vLLM tool parser development.

Tests what output format a model naturally produces with different
system prompt strategies:
  1. default - No format instruction (model's native behavior)
  2. hermes  - <tool_call> tag induction (hermes format)
  3. tools   - <tools> tag induction (our custom format)

Uses the same test queries from test_cases.yaml for consistency
with parser integration tests and unit tests.

Usage:
    uv run python test_output_format.py --model Qwen/Qwen2.5-7B-Instruct
    uv run python test_output_format.py --edge-only
    uv run python test_output_format.py --repeat 3
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import argparse
import asyncio
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

# Windows terminal UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace")


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_DEFAULT = """You are a helpful assistant with access to various tools.
When you need to use a tool, output the tool call in JSON format.

Available tools will be provided."""

SYSTEM_PROMPT_HERMES = """You are a helpful assistant with access to various tools.
When you need to use a tool, you MUST output the tool call wrapped in <tool_call> tags like this:

<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Example:
User: What's the weather in Seoul?
<tool_call>
{"name": "get_weather", "arguments": {"city": "Seoul"}}
</tool_call>

Available tools will be provided."""

SYSTEM_PROMPT_TOOLS = """You are a helpful assistant with access to various tools.
When you need to use a tool, you MUST output the tool call wrapped in <tools> tags like this:

<tools>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tools>

Example:
User: What's the weather in Seoul?
<tools>
{"name": "get_weather", "arguments": {"city": "Seoul"}}
</tools>

Available tools will be provided."""

PROMPT_CONFIGS = [
    ("default", SYSTEM_PROMPT_DEFAULT),
    ("hermes", SYSTEM_PROMPT_HERMES),
    ("tools", SYSTEM_PROMPT_TOOLS),
]


# =============================================================================
# Test Data Loaders
# =============================================================================

def load_tools(yaml_path: Path) -> list[dict[str, Any]]:
    """Load tool definitions from YAML in OpenAI format."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tools = []
    for tool in data.get("tools", []):
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
        })
    return tools


def load_queries(yaml_path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Load test queries from test_cases.yaml grouped by category.

    Returns: {category: [{query, description}, ...]}
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    result = {}
    for category in ["single", "parallel", "complex_args", "edge_cases", "no_tool"]:
        cases = data.get(category, [])
        result[category] = [
            {"query": c["query"], "description": c.get("description", "")}
            for c in cases
        ]
    return result


# =============================================================================
# Output Format Classification
# =============================================================================

def classify_output_format(content: str) -> dict:
    """
    Classify LLM output format.

    Returns:
        {
            "format": "tool_call_tag" | "tools_tag" | "json_codeblock"
                      | "plain_json" | "other" | "empty",
            "extracted_json": {...} | None,
            "raw": str
        }
    """
    if not content:
        return {"format": "empty", "extracted_json": None, "raw": ""}

    content = content.strip()
    result = {"raw": content, "extracted_json": None, "format": "other"}

    # 1. <tool_call> tag (hermes format)
    tool_call_match = re.search(
        r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    if tool_call_match:
        result["format"] = "tool_call_tag"
        try:
            result["extracted_json"] = json.loads(
                tool_call_match.group(1).strip())
        except json.JSONDecodeError:
            pass
        return result

    # 2. <tools> tag (our format)
    tools_match = re.search(
        r'<tools>\s*(.*?)\s*</tools>', content, re.DOTALL)
    if tools_match:
        result["format"] = "tools_tag"
        try:
            result["extracted_json"] = json.loads(
                tools_match.group(1).strip())
        except json.JSONDecodeError:
            pass
        return result

    # 3. JSON code block
    codeblock_match = re.search(
        r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if codeblock_match:
        result["format"] = "json_codeblock"
        try:
            result["extracted_json"] = json.loads(
                codeblock_match.group(1).strip())
        except json.JSONDecodeError:
            pass
        return result

    # 4. Plain JSON (tool call object in text)
    try:
        json_match = re.search(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if "name" in parsed:
                result["format"] = "plain_json"
                result["extracted_json"] = parsed
                return result
    except json.JSONDecodeError:
        pass

    return result


# =============================================================================
# Test Runner
# =============================================================================

async def run_single_query(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    query: str,
    tools: list,
) -> dict:
    """Run a single query and return raw result."""
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                tools=tools,
                temperature=0.0,
                max_tokens=1024,
            ),
            timeout=60.0,
        )

        message = response.choices[0].message

        # Native tool_calls (vLLM parsed successfully)
        if message.tool_calls:
            return {
                "query": query,
                "native_tool_calls": True,
                "tool_calls": [
                    {"name": tc.function.name,
                     "arguments": tc.function.arguments}
                    for tc in message.tool_calls
                ],
                "content": message.content,
                "format_analysis": None,
            }

        # Analyze content format
        format_analysis = classify_output_format(message.content or "")

        return {
            "query": query,
            "native_tool_calls": False,
            "tool_calls": None,
            "content": message.content,
            "format_analysis": format_analysis,
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        return {"query": query, "error": str(e)}


async def run_format_test(
    client: AsyncOpenAI,
    model: str,
    prompt_name: str,
    system_prompt: str,
    queries: list[str],
    tools: list,
) -> dict:
    """Run format test for a single prompt strategy."""
    print(f"\n{'='*60}")
    print(f"Test: {prompt_name}")
    print(f"{'='*60}")

    results = []
    format_counts: dict[str, int] = {}

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query[:60]}")
        result = await run_single_query(
            client, model, system_prompt, query, tools)
        results.append(result)

        if "error" in result:
            print(f"  X Error: {result['error']}")
            format_counts["error"] = format_counts.get("error", 0) + 1
        elif result["native_tool_calls"]:
            tool_name = result["tool_calls"][0]["name"]
            print(f"  [native] {tool_name}")
            format_counts["native"] = format_counts.get("native", 0) + 1
        else:
            fmt = result["format_analysis"]["format"]
            extracted = result["format_analysis"]["extracted_json"]
            if extracted and "name" in extracted:
                print(f"  [{fmt}] {extracted['name']}")
            else:
                raw = result["format_analysis"]["raw"][:80]
                print(f"  [{fmt}] {raw}")
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

    return {
        "prompt_name": prompt_name,
        "total": len(queries),
        "format_counts": format_counts,
        "results": results,
    }


async def run_edge_cases(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    queries_by_category: dict[str, list[dict]],
    tools: list,
) -> dict:
    """Run edge case tests (parallel, no_tool, complex_args)."""
    print(f"\n{'='*60}")
    print("Edge Case Tests")
    print(f"{'='*60}")

    results: dict[str, list] = {}

    for category in ["parallel", "complex_args"]:
        cases = queries_by_category.get(category, [])
        if not cases:
            continue

        results[category] = []
        for case in cases:
            query = case["query"]
            desc = case.get("description", "")
            print(f"\n[{category}] {desc or query[:50]}")

            result = await run_single_query(
                client, model, system_prompt, query, tools)
            result["category"] = category
            result["description"] = desc

            if "error" in result:
                print(f"  X Error: {result['error']}")
            elif result["native_tool_calls"]:
                count = len(result["tool_calls"])
                names = [tc["name"] for tc in result["tool_calls"]]
                print(f"  [native] {count} calls: {names}")
            else:
                fmt = result["format_analysis"]["format"]
                raw = result["format_analysis"]["raw"]

                if category == "parallel":
                    tools_count = raw.count("<tools>")
                    tool_call_count = raw.count("<tool_call>")
                    extracted = result["format_analysis"]["extracted_json"]
                    if isinstance(extracted, list):
                        print(f"  [{fmt}] array: {len(extracted)} items")
                    elif tools_count > 1:
                        print(f"  [{fmt}] {tools_count} <tools> tags")
                    elif tool_call_count > 1:
                        print(f"  [{fmt}] {tool_call_count} <tool_call> tags")
                    elif extracted:
                        print(f"  [{fmt}] single: {extracted.get('name')}")
                    else:
                        print(f"  [{fmt}] {raw[:60]}")

                elif category == "complex_args":
                    extracted = result["format_analysis"]["extracted_json"]
                    if extracted:
                        print("  OK JSON parsed")
                        if "arguments" in extracted:
                            args_preview = str(
                                extracted["arguments"])[:60]
                            print(f"     args: {args_preview}")
                    else:
                        print("  FAIL: JSON parse failed")

            results[category].append(result)

    return results


# =============================================================================
# Summary
# =============================================================================

def print_format_summary(all_results: list[dict]):
    """Print format distribution summary."""
    print(f"\n{'='*60}")
    print("Format Distribution Summary")
    print(f"{'='*60}")

    for suite in all_results:
        name = suite["prompt_name"]
        total = suite["total"]
        counts = suite["format_counts"]
        print(f"\n  {name}:")
        for fmt, count in sorted(counts.items()):
            pct = count / total * 100
            print(f"    {fmt}: {count}/{total} ({pct:.1f}%)")


def print_edge_summary(edge_results: dict[str, list]):
    """Print edge case summary."""
    print(f"\n{'='*60}")
    print("Edge Case Summary")
    print(f"{'='*60}")

    for category, cases in edge_results.items():
        success = 0
        for case in cases:
            if "error" in case:
                continue
            if category == "parallel":
                if case.get("native_tool_calls"):
                    if len(case["tool_calls"]) >= 2:
                        success += 1
                else:
                    raw = case.get("format_analysis", {}).get("raw", "")
                    if (raw.count("<tools>") >= 2
                            or raw.count("<tool_call>") >= 2):
                        success += 1
                    extracted = case.get(
                        "format_analysis", {}).get("extracted_json")
                    if isinstance(extracted, list) and len(extracted) >= 2:
                        success += 1
            elif category == "complex_args":
                if case.get("native_tool_calls"):
                    success += 1
                elif case.get("format_analysis", {}).get("extracted_json"):
                    success += 1
        print(f"\n  {category}: {success}/{len(cases)}")


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Output format analysis for vLLM tool parser development")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="LLM API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each query N times")
    parser.add_argument("--edge-only", action="store_true",
                        help="Run edge cases only")
    args = parser.parse_args()

    api_base = args.api_base
    model = args.model

    print(f"API Base: {api_base}")
    print(f"Model: {model}")

    client = AsyncOpenAI(base_url=api_base, api_key="dummy")

    # Load tools and queries from YAML
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    tools = load_tools(fixtures_dir / "test_tools.yaml")
    queries_by_category = load_queries(fixtures_dir / "test_cases.yaml")

    # Use "single" category for main format tests (clear tool-calling intent)
    base_queries = [c["query"] for c in queries_by_category.get("single", [])]
    queries = base_queries * args.repeat

    print(f"Tools: {len(tools)}")
    print(f"Queries: {len(queries)} ({len(base_queries)} x {args.repeat})")

    results_data: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model,
        "repeat": args.repeat,
    }

    # Main format tests
    if not args.edge_only:
        all_format_results = []
        for prompt_name, system_prompt in PROMPT_CONFIGS:
            suite_result = await run_format_test(
                client, model, prompt_name, system_prompt, queries, tools)
            all_format_results.append(suite_result)
            results_data[f"{prompt_name}_results"] = suite_result

        print_format_summary(all_format_results)

    # Edge case tests (use "tools" prompt for edge cases)
    edge_results = await run_edge_cases(
        client, model, SYSTEM_PROMPT_TOOLS, queries_by_category, tools)
    results_data["edge_case_results"] = edge_results
    print_edge_summary(edge_results)

    # Save results
    timestamp = results_data["timestamp"]
    output_file = (
        Path(__file__).parent.parent / "results"
        / f"output_format_{timestamp}.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
