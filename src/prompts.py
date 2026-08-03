"""Prompts used by the domain-independent SimulRAG pipeline."""

from __future__ import annotations

import json
from typing import Any


def initial_answer_prompt(question: str) -> str:
    return f"""Please provide a comprehensive answer to the following scientific question.
Focus on factual, well-reasoned information. Do not call or assume access to an external
simulator; answer from your existing knowledge.

Question: {question}

Answer:"""


def claim_decomposition_prompt(answer: str) -> str:
    return f"""Please deconstruct the following paragraph into the smallest possible standalone self-contained facts without semantic repetition, and return a JSON array where each item is {{"claim": "..."}}.

CRITICAL: Extract ONLY the 8-12 MOST IMPORTANT claims. Be extremely selective. Focus ONLY on:
- Direct answers to the specific question asked
- Specific numerical values, percentages, or measurements
- Key causal relationships (A causes B)
- Critical scientific conclusions

STRICTLY AVOID:
- General background information
- Basic definitions
- Procedural explanations
- Location descriptions
- Any claim that does not directly address the question

Each claim must be essential to answering the question. If unsure whether to include a claim, do not include it.

The input is:
{answer}"""


def merge_claims_prompt(
    existing_claims: list[dict[str, Any]], new_claims: list[dict[str, Any]]
) -> str:
    existing = "\n".join(
        f"{index}: {item['claim']}" for index, item in enumerate(existing_claims)
    )
    new = "\n".join(
        f"{index}: {item['claim']}" for index, item in enumerate(new_claims)
    )
    return f"""You are given two sets of claims. Find which claims in Set B are already covered by claims in Set A.

Set A (existing claims):
{existing}

Set B (new claims):
{new}

For each claim in Set B, check whether it is semantically equivalent to a claim in Set A. Opposite trends or incompatible numerical values are not equivalent.

Return ONLY a valid JSON array of pairs. Each pair is [existing_index, new_index] where Set A[existing_index] covers Set B[new_index]. Return [] if none match."""


def entailment_prompt(answer: str, claims: list[dict[str, Any]]) -> str:
    claim_text = "\n".join(
        f"{index}: {item['claim']}" for index, item in enumerate(claims)
    )
    return f"""Determine which standalone claims are semantically supported by the answer.

ANSWER:
{answer}

CLAIMS:
{claim_text}

Judge every answer-claim pair. A claim is supported only when the answer entails the same factual meaning; shared topic or plausibility alone is insufficient. Opposite trends, incompatible quantities, or missing qualifications are not support.

Return only JSON with the indices of supported claims:
{{"supported_claim_ids": [0, 2]}}"""


def boundary_prompt(question: str, claims: list[dict[str, Any]], handbook: str) -> str:
    claim_text = "\n".join(
        f"{index}: {item['claim']}" for index, item in enumerate(claims)
    )
    return f"""You are an expert in scientific simulation. Determine whether each claim can be directly and comprehensively verified using the simulator described in the handbook.

SIMULATOR HANDBOOK:
{handbook}

QUESTION:
{question}

CLAIMS:
{claim_text}

A claim is verifiable only if the simulator's available inputs and outputs can directly assess it. Verification may either support or refute the claim, and required inputs may be supplied by the original question. Therefore, mark a claim verifiable when it says a simulator-derived value is unknown or unavailable but the described interface can compute that value. General policy, normative, explanatory, or out-of-scope claims are not verifiable.

Return ONLY this JSON structure, preserving every claim_id:
{{"claims": [{{"claim_id": 0, "verifiable": true, "reason": "brief reason"}}]}}"""


def tool_call_prompt(question: str, selected_claims: list[str], handbook: str) -> str:
    claims = "\n".join(f"- {claim}" for claim in selected_claims)
    return f"""You are an expert AI system that translates natural language questions about scientific forecasting into structured parameters for a simulation tool.

Your task:
Given an open-ended scientific question and the claims selected for verification, extract the precise parameters needed to run the simulator.

Tool handbook:
{handbook}

Question:
{question}

Claims selected for verification:
{claims}

Output format: Return a single JSON object with these exact top-level keys:
{{"tool": "exact function name from the handbook", "arguments": {{"exact_parameter_name": "value"}}}}

Use only function and parameter names defined in the handbook. Respect all stated units, valid ranges, required fields, and types. Return JSON only."""


def claim_update_prompt(claim: str, evidence: dict[str, Any]) -> str:
    evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
    return f"""You are a fact-checking assistant. Analyze whether trusted simulator evidence is relevant to a claim and whether the claim should be minimally updated for accuracy.

INSTRUCTIONS:
1. Set is_included to true only when the evidence directly concerns the claim.
2. Set should_update to true only when the claim is incorrect or incomplete and the evidence can correct it.
3. If updating, modify only the related part of the claim; do not introduce unrelated claims.
4. Carefully distinguish absolute values, differences, directions, units, and baselines. Perform any required numerical calculation.
5. If evidence is insufficient, set is_included and should_update to false.
6. If the claim says a value is unknown, unavailable, or cannot be computed but the evidence supplies that value, set is_included and should_update to true and replace the claim with the evidence-supported result.

CLAIM:
{claim}

TRUSTED SIMULATOR EVIDENCE:
{evidence_text}

Return only JSON:
{{"is_included": true, "should_update": false, "updated_claim": "claim text"}}"""


def final_answer_prompt(question: str, claims: list[str]) -> str:
    claim_text = "\n".join(f"- {claim}" for claim in claims)
    return f"""You are an expert answering a complex scientific question from provided factual claims.

QUESTION: {question}

AVAILABLE CLAIMS:
{claim_text}

Generate a comprehensive and accurate answer using only the information in the claims. Synthesize them into a coherent response, prioritize specific evidence when claims conflict, acknowledge material limitations, and do not add unsupported information."""


def final_answer_evaluation_prompt(
    question: str, generated_answer: str, reference_answer: str
) -> str:
    return f"""Evaluate a generated scientific answer against a reference answer.

QUESTION: {question}

GENERATED ANSWER: {generated_answer}

REFERENCE ANSWER: {reference_answer}

Judge semantic correctness and relevance, not exact wording. Do not require exact numerical matching. For temperatures, an absolute discrepancy of at most 0.05 degrees Celsius is a minor rounding difference. If the only discrepancy is minor rounding and it does not alter the scientific conclusion or magnitude category, you must set is_correct to true and assign a score of at least 0.8. Return only JSON:
{{"is_correct": true, "score": 0.0, "reason": "brief explanation"}}

The score is from 0 to 1. Mark is_correct false for contradictions, significant factual errors, or missing information that changes the conclusion."""


def claim_evaluation_prompt(question: str, claim: str, reference_answer: str) -> str:
    return f"""Evaluate whether a generated atomic claim is correct and relevant to the question using the reference answer as evidence.

QUESTION: {question}

CLAIM: {claim}

REFERENCE ANSWER: {reference_answer}

A claim is true only when it is both factually correct and relevant. Return only JSON:
{{"is_correct": true, "is_relevant": true, "reason": "brief explanation"}}"""
