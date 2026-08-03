"""LLM-based final-answer and atomic-claim evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .data_io import load_dataset, write_json
from .llm_helper import LLMClient
from .prompts import claim_evaluation_prompt, final_answer_evaluation_prompt


async def _evaluate_claim(
    llm: LLMClient,
    question: str,
    claim: dict[str, Any],
    reference: str,
    temperature: float,
) -> dict[str, Any]:
    raw = await llm.complete_json(
        claim_evaluation_prompt(question, claim["final_claim"], reference),
        temperature=temperature,
        max_tokens=500,
    )
    if not isinstance(raw, dict):
        raise TypeError("Claim judge returned a non-object")
    return {
        "claim_id": claim["claim_id"],
        "claim": claim["final_claim"],
        "is_correct": bool(raw.get("is_correct", False)),
        "is_relevant": bool(raw.get("is_relevant", False)),
        "reason": str(raw.get("reason", "")),
    }


async def evaluate_run(
    run_dir: Path,
    llm: LLMClient,
    *,
    temperature: float = 0.1,
) -> list[dict[str, Any]]:
    final_path = run_dir / "06_final_output.json"
    records = json.loads(final_path.read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    dataset = load_dataset(run_config["dataset_path"])
    references = {
        item["question"]: item["reference_answer"] for item in dataset.records
    }
    results: list[dict[str, Any]] = []

    for record in records:
        if "final_answer" not in record:
            results.append(
                {
                    "question_index": record.get("question_index"),
                    "error": "Pipeline did not produce a final answer",
                }
            )
            continue
        question = record["question"]
        dataset_index = record.get("dataset_index")
        if isinstance(dataset_index, int) and 0 <= dataset_index < len(dataset.records):
            source_record = dataset.records[dataset_index]
            reference = (
                source_record["reference_answer"]
                if source_record["question"] == question
                else references.get(question)
            )
        else:
            reference = references.get(question)
        if not reference:
            results.append(
                {
                    "question_index": record["question_index"],
                    "error": "No reference answer found for this question",
                }
            )
            continue
        answer_task = llm.complete_json(
            final_answer_evaluation_prompt(question, record["final_answer"], reference),
            temperature=temperature,
            max_tokens=600,
        )
        claim_tasks = [
            _evaluate_claim(llm, question, claim, reference, temperature)
            for claim in record.get("final_claims", [])
        ]
        judged = await asyncio.gather(answer_task, *claim_tasks, return_exceptions=True)
        answer_raw = judged[0]
        if isinstance(answer_raw, BaseException) or not isinstance(answer_raw, dict):
            answer_result = {
                "is_correct": False,
                "score": 0.0,
                "reason": f"Evaluation failed: {answer_raw}",
            }
        else:
            try:
                score = max(0.0, min(1.0, float(answer_raw.get("score", 0.0))))
            except (TypeError, ValueError):
                score = 0.0
            answer_result = {
                "is_correct": bool(answer_raw.get("is_correct", False)),
                "score": score,
                "reason": str(answer_raw.get("reason", "")),
            }

        claim_results: list[dict[str, Any]] = []
        for claim, judged_claim in zip(record.get("final_claims", []), judged[1:]):
            if isinstance(judged_claim, BaseException):
                claim_results.append(
                    {
                        "claim_id": claim["claim_id"],
                        "claim": claim["final_claim"],
                        "is_correct": False,
                        "is_relevant": False,
                        "error": str(judged_claim),
                    }
                )
            else:
                claim_results.append(judged_claim)
        true_claims = sum(
            item.get("is_correct", False) and item.get("is_relevant", False)
            for item in claim_results
        )
        result = {
            "question_index": record["question_index"],
            "question": question,
            "answer_evaluation": answer_result,
            "claim_evaluations": claim_results,
            "claim_accuracy": true_claims / len(claim_results)
            if claim_results
            else 0.0,
        }
        results.append(result)
        question_dir = run_dir / record["artifact_dir"]
        write_json(question_dir / "07_evaluation.json", result)

    write_json(run_dir / "07_evaluation.json", results)
    return results
