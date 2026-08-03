"""Create open scientific QA data from simulator-derived question/answer pairs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .data_io import DATA_ROOT, PROJECT_ROOT, write_json
from .llm_helper import LLMClient
from .simulator import CustomSimulator


def normalize_existing_record(
    item: dict[str, Any], index: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Split a legacy benchmark item into public data and simulator evidence."""
    question = item.get("question") or item.get("open_question")
    reference = item.get("reference_answer")
    derived_answer = item.get("derived_quantitative_answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Input item {index} has no question")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Input item {index} has no reference_answer")
    if not isinstance(derived_answer, str) or not derived_answer.strip():
        raise ValueError(f"Input item {index} has no derived_quantitative_answer")
    derived_question = item.get("derived_quantitative_question") or question
    return (
        {
            "question": question.strip(),
            "reference_answer": reference.strip(),
        },
        {
            "question": question.strip(),
            "derived_quantitative_question": str(derived_question).strip(),
            "derived_quantitative_answer": derived_answer.strip(),
        },
    )


def generation_prompt(item: dict[str, Any], variation: int) -> str:
    quantitative_question = item.get("derived_quantitative_question") or item.get(
        "question"
    )
    quantitative_answer = item.get("derived_quantitative_answer") or item.get("answer")
    topic = item.get("topic", "scientific decision-making")
    return f"""Create one realistic, open-ended scientific question and its reference answer from trusted simulator evidence.

TOPIC: {topic}
SIMULATOR QUESTION: {quantitative_question}
SIMULATOR ANSWER: {quantitative_answer}
VARIATION ID: {variation}

The open question should require interpreting the simulator result in scientific context rather than copying the quantitative question. The reference answer must be fully supported by the simulator evidence, state important numerical results accurately, and avoid unsupported facts. Produce a distinct wording and emphasis for each variation.

Return only JSON:
{{"question": "...", "reference_answer": "..."}}"""


async def generate_record(
    llm: LLMClient,
    item: dict[str, Any],
    variation: int,
    temperature: float,
) -> tuple[dict[str, str], dict[str, str]]:
    raw = await llm.complete_json(
        generation_prompt(item, variation),
        temperature=temperature,
        max_tokens=1600,
    )
    if not isinstance(raw, dict):
        raise TypeError("Data generator returned a non-object")
    question = raw.get("question")
    reference = raw.get("reference_answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question is empty")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Generated reference_answer is empty")
    public = {
        "question": question.strip(),
        "reference_answer": reference.strip(),
    }
    cache = {
        "question": question.strip(),
        "derived_quantitative_question": str(
            item.get("derived_quantitative_question", item.get("question", ""))
        ),
        "derived_quantitative_answer": str(
            item.get("derived_quantitative_answer", item.get("answer", ""))
        ),
    }
    return public, cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn simulator-derived QA pairs into open questions and references."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON array containing derived_quantitative_question/answer pairs.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Public questions.json output."
    )
    parser.add_argument(
        "--cache-output",
        type=Path,
        help="Optional simulator_evidence.json output for replay.",
    )
    parser.add_argument("--variations", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of input records to process.",
    )
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Split legacy open_question benchmark JSON without LLM generation.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip legacy records that lack required public or evidence fields.",
    )
    parser.add_argument(
        "--handbook",
        type=Path,
        help="Simulator handbook for live generation when input answers are absent.",
    )
    parser.add_argument(
        "--simulator",
        help="Simulator script/command/URL for live generation when answers are absent.",
    )
    parser.add_argument(
        "--provider", choices=("openai", "anthropic", "gemini"), default="openai"
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    return parser


async def run(args: argparse.Namespace) -> None:
    if args.variations < 1:
        raise ValueError("--variations must be positive")
    if args.input.is_absolute():
        input_path = args.input
    else:
        data_path = DATA_ROOT / args.input
        input_path = data_path if data_path.exists() else PROJECT_ROOT / args.input
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise TypeError("Input must be a JSON array")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.normalize_existing:
        generated = []
        skipped: list[tuple[int, str]] = []
        for index, item in enumerate(source):
            if not isinstance(item, dict):
                raise TypeError(f"Input item {index} must be an object")
            try:
                generated.append(normalize_existing_record(item, index))
            except ValueError as exc:
                if not args.skip_incomplete:
                    raise
                skipped.append((index, str(exc)))
        if skipped:
            print(
                f"Skipped {len(skipped)} incomplete records at source indices "
                + ", ".join(str(index) for index, _ in skipped)
            )
        if args.limit is not None and len(generated) > args.limit:
            print(f"Using the first {args.limit} of {len(generated)} valid records")
            generated = generated[: args.limit]
        public = [item[0] for item in generated]
        cache = [item[1] for item in generated]
        output = args.output if args.output.is_absolute() else DATA_ROOT / args.output
        write_json(output, public)
        if not args.cache_output:
            raise ValueError("--normalize-existing requires --cache-output")
        cache_output = (
            args.cache_output
            if args.cache_output.is_absolute()
            else DATA_ROOT / args.cache_output
        )
        write_json(cache_output, cache)
        print(f"Normalized {len(public)} questions to {output}")
        return

    if args.limit is not None:
        if len(source) > args.limit:
            print(f"Using the first {args.limit} of {len(source)} input records")
        source = source[: args.limit]

    needs_simulation = False
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise TypeError(f"Input item {index} must be an object")
        if not (item.get("derived_quantitative_question") or item.get("question")):
            raise ValueError(f"Input item {index} lacks a quantitative question")
        if not (item.get("derived_quantitative_answer") or item.get("answer")):
            needs_simulation = True
    if needs_simulation and not (args.handbook and args.simulator):
        raise ValueError(
            "Items without simulator answers require --handbook and --simulator"
        )

    llm = LLMClient(
        args.provider,
        args.model,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
    )
    if needs_simulation:
        handbook_path = (
            args.handbook
            if args.handbook.is_absolute()
            else (PROJECT_ROOT / args.handbook).resolve()
        )
        handbook = handbook_path.read_text(encoding="utf-8")
        simulator = CustomSimulator(args.simulator)

        async def add_evidence(item: dict[str, Any]) -> dict[str, Any]:
            if item.get("derived_quantitative_answer") or item.get("answer"):
                return item
            quantitative_question = str(
                item.get("derived_quantitative_question") or item["question"]
            )
            result = await simulator.query(
                quantitative_question,
                [quantitative_question],
                llm,
                handbook,
                0.1,
                args.max_retries,
            )
            return {
                **item,
                "derived_quantitative_question": quantitative_question,
                "derived_quantitative_answer": json.dumps(result, ensure_ascii=False),
            }

        source = list(await asyncio.gather(*(add_evidence(item) for item in source)))

    tasks = [
        generate_record(llm, item, variation, args.temperature)
        for item in source
        for variation in range(args.variations)
    ]
    generated = await asyncio.gather(*tasks)
    public = [item[0] for item in generated]
    cache = [item[1] for item in generated]
    output = args.output if args.output.is_absolute() else DATA_ROOT / args.output
    write_json(output, public)
    if args.cache_output:
        cache_output = (
            args.cache_output
            if args.cache_output.is_absolute()
            else DATA_ROOT / args.cache_output
        )
        write_json(cache_output, cache)
    print(f"Wrote {len(public)} questions to {output}")


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
