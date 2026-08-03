#!/usr/bin/env python3
"""Run SimulRAG on a built-in or custom scientific QA dataset."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.config import CENTRALITY_METRICS, PipelineConfig
from src.data_io import (
    DATA_ROOT,
    PROJECT_ROOT,
    create_run_dir,
    load_builtin_evidence,
    load_dataset,
)
from src.llm_helper import LLMClient
from src.pipeline import SimulRAGPipeline
from src.simulator import CachedSimulator, CustomSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run simulator-grounded claim verification and answer generation."
    )
    parser.add_argument(
        "--dataset",
        default="climate",
        help="Built-in alias (climate, epi, urban) or a custom JSON path.",
    )
    parser.add_argument(
        "--data",
        help="Custom dataset path, relative to data/ or the project root.",
    )
    parser.add_argument("--start", type=int, default=0, help="First question index.")
    parser.add_argument("--count", type=int, default=1, help="Number of questions.")
    parser.add_argument("--run-name", help="Optional artifact-directory label.")

    parser.add_argument(
        "--handbook",
        help="Custom simulator handbook path. Required for custom datasets.",
    )
    parser.add_argument(
        "--simulator",
        "--simulator-path",
        dest="simulator",
        help="Custom executable/Python script, command with arguments, or HTTP endpoint.",
    )

    parser.add_argument(
        "--provider", choices=("openai", "anthropic", "gemini"), default="openai"
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("-m", "--num-answers", type=int, default=5)
    parser.add_argument(
        "--centrality",
        "--uncertainty-metric",
        choices=tuple(CENTRALITY_METRICS),
        default="closeness",
        help="Graph centrality used for tau selection and kappa filtering.",
    )
    parser.add_argument(
        "--tau", type=float, default=0.4, help="UE selection threshold."
    )
    parser.add_argument(
        "--kappa", type=float, default=0.4, help="Final claim-filter threshold."
    )
    parser.add_argument("--answer-temperature", type=float, default=1.0)
    parser.add_argument("--decomposition-temperature", type=float, default=0.1)
    parser.add_argument("--boundary-temperature", type=float, default=0.1)
    parser.add_argument("--tool-temperature", type=float, default=0.1)
    parser.add_argument("--update-temperature", type=float, default=0.1)
    parser.add_argument("--final-temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    return parser


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    data_path = (DATA_ROOT / path).resolve()
    if data_path.exists():
        return data_path
    return (PROJECT_ROOT / path).resolve()


async def run(args: argparse.Namespace) -> Path:
    if args.start < 0 or args.count < 1:
        raise ValueError("--start must be nonnegative and --count must be positive")
    dataset = load_dataset(args.dataset, args.data)
    selected_records = dataset.records[args.start : args.start + args.count]
    records = [
        {"question": record["question"], "_dataset_index": args.start + offset}
        for offset, record in enumerate(selected_records)
    ]
    if not records:
        raise ValueError(
            f"No questions selected from {dataset.path}; start={args.start}, count={args.count}"
        )

    config = PipelineConfig(
        provider=args.provider,
        model=args.model,
        num_answers=args.num_answers,
        centrality=args.centrality,
        selection_threshold=args.tau,
        filtering_threshold=args.kappa,
        answer_temperature=args.answer_temperature,
        decomposition_temperature=args.decomposition_temperature,
        boundary_temperature=args.boundary_temperature,
        tool_temperature=args.tool_temperature,
        update_temperature=args.update_temperature,
        final_temperature=args.final_temperature,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
    )
    config.validate()
    llm = LLMClient(
        config.provider,
        config.model,
        max_retries=config.max_retries,
        concurrency=config.concurrency,
    )

    if dataset.builtin:
        handbook_path = DATA_ROOT / "handbooks" / f"{dataset.name}.md"
        simulator = CachedSimulator(load_builtin_evidence(dataset.name))
    else:
        if not args.handbook or not args.simulator:
            raise ValueError("Custom datasets require both --handbook and --simulator")
        handbook_path = resolve_path(args.handbook)
        simulator = CustomSimulator(args.simulator)
    handbook = handbook_path.read_text(encoding="utf-8")

    run_dir = create_run_dir(dataset.name, args.run_name)
    pipeline = SimulRAGPipeline(
        dataset=dataset,
        config=config,
        llm=llm,
        simulator=simulator,
        handbook=handbook,
        run_dir=run_dir,
    )
    results = await pipeline.run(records)
    successful = sum("final_answer" in result for result in results)
    print(f"Completed {successful}/{len(results)} questions")
    print(f"Artifacts: {run_dir}")
    if successful == 0:
        raise RuntimeError(
            "No question completed successfully; inspect the run artifacts"
        )
    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
