#!/usr/bin/env python3
"""Evaluate a completed SimulRAG run."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.data_io import latest_run
from src.evaluation import evaluate_run
from src.llm_helper import LLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate final answers and claims.")
    parser.add_argument(
        "--run-dir", type=Path, help="Completed run artifact directory."
    )
    parser.add_argument(
        "--dataset", help="Use the latest completed run for this dataset alias."
    )
    parser.add_argument(
        "--provider", choices=("openai", "anthropic", "gemini"), default="openai"
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    return parser


async def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run(args.dataset)
    llm = LLMClient(
        args.provider,
        args.model,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
    )
    results = await evaluate_run(run_dir, llm, temperature=args.temperature)
    valid = [item for item in results if "answer_evaluation" in item]
    correct = sum(item["answer_evaluation"]["is_correct"] for item in valid)
    mean_claim_accuracy = (
        sum(item["claim_accuracy"] for item in valid) / len(valid) if valid else 0.0
    )
    print(f"Answer correctness: {correct}/{len(valid)}")
    print(f"Mean claim accuracy: {mean_claim_accuracy:.3f}")
    print(f"Evaluation: {run_dir / '07_evaluation.json'}")


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
