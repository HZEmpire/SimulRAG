"""End-to-end SimulRAG claim generation and verification pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import networkx as nx

from .config import PipelineConfig
from .data_io import Dataset, write_json
from .llm_helper import LLMClient
from .prompts import (
    boundary_prompt,
    claim_decomposition_prompt,
    claim_update_prompt,
    entailment_prompt,
    final_answer_prompt,
    initial_answer_prompt,
    merge_claims_prompt,
)


class SimulRAGPipeline:
    def __init__(
        self,
        *,
        dataset: Dataset,
        config: PipelineConfig,
        llm: LLMClient,
        simulator: Any,
        handbook: str,
        run_dir: Path,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.llm = llm
        self.simulator = simulator
        self.handbook = handbook
        self.run_dir = run_dir

    async def _resilient_gather(
        self, calls: list[Awaitable[Any]], *, stage: str
    ) -> list[Any]:
        results = await asyncio.gather(*calls, return_exceptions=True)
        successful: list[Any] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                print(f"  {stage} item {index} failed after retries: {result}")
            else:
                successful.append(result)
        return successful

    async def _generate_answers(self, question: str) -> list[str]:
        prompt = initial_answer_prompt(question)
        calls = [
            self.llm.complete(
                prompt,
                temperature=self.config.answer_temperature,
                max_tokens=1800,
            )
            for _ in range(self.config.num_answers)
        ]
        answers = await self._resilient_gather(calls, stage="initial answer")
        if not answers:
            raise RuntimeError("No initial answer could be generated")
        return [str(answer).strip() for answer in answers if str(answer).strip()]

    async def _decompose_answer(self, answer: str) -> list[dict[str, Any]]:
        raw = await self.llm.complete_json(
            claim_decomposition_prompt(answer),
            temperature=self.config.decomposition_temperature,
            max_tokens=1800,
        )
        if isinstance(raw, dict):
            raw = raw.get("claims", [])
        if not isinstance(raw, list):
            raise TypeError("Claim decomposition must return a list")

        claims: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("claim")
            if not isinstance(text, str) or len(text.strip()) < 5:
                continue
            value = item.get("confidence", item.get("gpt-confidence", 0.5))
            try:
                confidence = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                confidence = 0.5
            claims.append({"claim": text.strip(), "verbalized_confidence": confidence})
        if not claims:
            raise ValueError("Claim decomposition returned no usable claims")
        return claims[:15]

    async def _decompose_answers(self, answers: list[str]) -> list[dict[str, Any]]:
        raw_results = await asyncio.gather(
            *(self._decompose_answer(answer) for answer in answers),
            return_exceptions=True,
        )
        results: list[dict[str, Any]] = []
        for index, (answer, result) in enumerate(zip(answers, raw_results)):
            if isinstance(result, BaseException):
                print(
                    f"  claim decomposition {index} failed: {result}; using answer fallback"
                )
                claims = [{"claim": answer, "verbalized_confidence": 0.5}]
            else:
                claims = result
            results.append({"answer_id": index, "answer": answer, "claims": claims})
        return results

    async def _merge_claims(
        self, decompositions: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
        merged = [dict(item) for item in decompositions[0]["claims"]]
        mapping: dict[int, list[int]] = {index: [0] for index in range(len(merged))}

        for answer_id, decomposition in enumerate(decompositions[1:], start=1):
            incoming = decomposition["claims"]
            try:
                pairs = await self.llm.complete_json(
                    merge_claims_prompt(merged, incoming),
                    temperature=self.config.decomposition_temperature,
                    max_tokens=1000,
                )
            except Exception as exc:  # noqa: BLE001 - retain claims after provider errors.
                print(
                    f"  claim merge {answer_id} failed: {exc}; treating claims as new"
                )
                pairs = []
            matched: dict[int, int] = {}
            if isinstance(pairs, list):
                for pair in pairs:
                    if (
                        isinstance(pair, list)
                        and len(pair) == 2
                        and all(isinstance(value, int) for value in pair)
                        and 0 <= pair[0] < len(merged)
                        and 0 <= pair[1] < len(incoming)
                    ):
                        matched.setdefault(pair[1], pair[0])
            for new_index, claim in enumerate(incoming):
                existing_index = matched.get(new_index)
                if existing_index is not None:
                    if answer_id not in mapping[existing_index]:
                        mapping[existing_index].append(answer_id)
                else:
                    merged.append(dict(claim))
                    mapping[len(merged) - 1] = [answer_id]
        return merged, mapping

    async def _build_entailment_mapping(
        self,
        answers: list[str],
        claims: list[dict[str, Any]],
        provenance: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        """Judge every answer-claim pair, batching claims once per answer."""
        raw_results = await asyncio.gather(
            *(
                self.llm.complete_json(
                    entailment_prompt(answer, claims),
                    temperature=self.config.decomposition_temperature,
                    max_tokens=1200,
                )
                for answer in answers
            ),
            return_exceptions=True,
        )
        mapping: dict[int, list[int]] = {index: [] for index in range(len(claims))}
        for answer_id, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                print(
                    f"  entailment judging for answer {answer_id} failed: {result}; "
                    "using merge provenance"
                )
                supported = [
                    claim_id
                    for claim_id, answer_ids in provenance.items()
                    if answer_id in answer_ids
                ]
            else:
                values = (
                    result.get("supported_claim_ids", [])
                    if isinstance(result, dict)
                    else result
                )
                supported = [
                    value
                    for value in values
                    if isinstance(value, int) and 0 <= value < len(claims)
                ]
            for claim_id in set(supported):
                mapping[claim_id].append(answer_id)
        return mapping

    @staticmethod
    def _pagerank(
        graph: nx.Graph,
        *,
        alpha: float = 0.85,
        tolerance: float = 1e-8,
        max_iterations: int = 200,
    ) -> dict[str, float]:
        """Small dependency-free PageRank implementation for the claim graph."""
        nodes = list(graph)
        if not nodes:
            return {}
        count = len(nodes)
        ranks = {node: 1.0 / count for node in nodes}
        for _ in range(max_iterations):
            dangling = sum(ranks[node] for node in nodes if graph.degree(node) == 0)
            updated = {
                node: (1.0 - alpha) / count + alpha * dangling / count for node in nodes
            }
            for source in nodes:
                degree = graph.degree(source)
                if degree:
                    contribution = alpha * ranks[source] / degree
                    for target in graph.neighbors(source):
                        updated[target] += contribution
            difference = sum(abs(updated[node] - ranks[node]) for node in nodes)
            ranks = updated
            if difference < tolerance:
                break
        return ranks

    @staticmethod
    def _add_graph_metrics(
        answers: list[str],
        claims: list[dict[str, Any]],
        mapping: dict[int, list[int]],
    ) -> dict[str, Any]:
        graph = nx.Graph()
        for answer_id in range(len(answers)):
            graph.add_node(f"answer_{answer_id}", bipartite=0)
        for claim_id in range(len(claims)):
            graph.add_node(f"claim_{claim_id}", bipartite=1)
            for answer_id in mapping.get(claim_id, []):
                graph.add_edge(f"answer_{answer_id}", f"claim_{claim_id}")

        metric_functions = {
            "closeness_centrality": nx.closeness_centrality,
            "degree_centrality": nx.degree_centrality,
            "betweenness_centrality": nx.betweenness_centrality,
            "eigenvalue_centrality": lambda value: nx.eigenvector_centrality(
                value, max_iter=5000
            ),
            "pagerank": SimulRAGPipeline._pagerank,
        }
        computed: dict[str, dict[str, float]] = {}
        for name, function in metric_functions.items():
            try:
                computed[name] = function(graph)
            except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
                computed[name] = {}

        for claim_id, claim in enumerate(claims):
            node = f"claim_{claim_id}"
            claim["claim_id"] = claim_id
            claim["answer_support"] = mapping.get(claim_id, [])
            claim["uncertainty_metrics"] = {
                name: float(values.get(node, 0.0)) for name, values in computed.items()
            }
            claim["uncertainty_metrics"]["verbalized_confidence"] = float(
                claim.get("verbalized_confidence", 0.5)
            )
        return {
            "response_nodes": len(answers),
            "claim_nodes": len(claims),
            "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph) if graph else 0,
        }

    async def _analyze_boundaries(
        self, question: str, claims: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # SBA is defined per atomic claim; calls are concurrent, while simulator
        # retrieval remains a single query for all selected claims.
        results = await asyncio.gather(
            *(
                self.llm.complete_json(
                    boundary_prompt(question, [claim], self.handbook),
                    temperature=self.config.boundary_temperature,
                    max_tokens=700,
                )
                for claim in claims
            ),
            return_exceptions=True,
        )
        analyzed: list[dict[str, Any]] = []
        for claim, result in zip(claims, results):
            item: dict[str, Any] = {}
            if isinstance(result, BaseException):
                print(
                    f"  boundary analysis for claim {claim['claim_id']} failed: "
                    f"{result}; marking it unverifiable"
                )
            elif isinstance(result, dict):
                items = result.get("claims", [])
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    item = items[0]
            analyzed.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim": claim["claim"],
                    "verifiable": bool(item.get("verifiable", False)),
                    "reason": str(item.get("reason", "")),
                }
            )
        return analyzed

    async def _update_claim(
        self, claim: dict[str, Any], evidence: dict[str, Any]
    ) -> dict[str, Any]:
        raw = await self.llm.complete_json(
            claim_update_prompt(claim["claim"], evidence),
            temperature=self.config.update_temperature,
            max_tokens=800,
        )
        if not isinstance(raw, dict):
            raise TypeError("Claim update must return an object")
        included = bool(raw.get("is_included", False))
        should_update = included and bool(raw.get("should_update", False))
        updated = raw.get("updated_claim", claim["claim"])
        if not isinstance(updated, str) or not updated.strip():
            updated = claim["claim"]
            should_update = False
        return {
            "claim_id": claim["claim_id"],
            "original_claim": claim["claim"],
            "is_included": included,
            "should_update": should_update,
            "updated_claim": updated.strip(),
        }

    async def process_question(
        self, record: dict[str, Any], question_index: int
    ) -> dict[str, Any]:
        question_dir = self.run_dir / "questions" / f"{question_index:04d}"
        question_dir.mkdir(parents=True, exist_ok=True)
        question = record["question"]
        dataset_index = int(record.get("_dataset_index", question_index))
        write_json(
            question_dir / "00_input.json",
            {
                "question_index": question_index,
                "dataset_index": dataset_index,
                "question": question,
            },
        )

        print(
            f"[{question_index}] generating {self.config.num_answers} initial answers"
        )
        answers = await self._generate_answers(question)
        write_json(question_dir / "01_initial_answers.json", {"answers": answers})

        print(f"[{question_index}] decomposing and merging claims")
        decompositions = await self._decompose_answers(answers)
        claims, provenance = await self._merge_claims(decompositions)
        entailment_mapping = await self._build_entailment_mapping(
            answers, claims, provenance
        )
        graph_stats = self._add_graph_metrics(answers, claims, entailment_mapping)
        write_json(
            question_dir / "02_claims.json",
            {
                "decompositions": decompositions,
                "merged_claims": claims,
                "merge_provenance": {
                    str(key): value for key, value in provenance.items()
                },
                "entailment_mapping": {
                    str(key): value for key, value in entailment_mapping.items()
                },
                "graph_stats": graph_stats,
            },
        )

        print(f"[{question_index}] analyzing simulator boundaries")
        boundary = await self._analyze_boundaries(question, claims)
        boundary_by_id = {item["claim_id"]: item for item in boundary}
        selected = [
            claim
            for claim in claims
            if boundary_by_id[claim["claim_id"]]["verifiable"]
            and claim["uncertainty_metrics"][self.config.centrality_key]
            < self.config.selection_threshold
        ]
        write_json(
            question_dir / "03_boundary_analysis.json",
            {
                "centrality": self.config.centrality,
                "centrality_key": self.config.centrality_key,
                "selection_threshold": self.config.selection_threshold,
                "claims": boundary,
                "selected_claim_ids": [claim["claim_id"] for claim in selected],
            },
        )

        evidence: dict[str, Any] = {
            "source": "not-called",
            "reason": "no selected claims",
        }
        if selected:
            print(
                f"[{question_index}] retrieving simulator evidence for {len(selected)} claims"
            )
            evidence = await self.simulator.query(
                question,
                [claim["claim"] for claim in selected],
                self.llm,
                self.handbook,
                self.config.tool_temperature,
                self.config.max_retries,
            )
        write_json(question_dir / "04_simulator_output.json", evidence)

        print(f"[{question_index}] updating selected claims")
        raw_updates = await asyncio.gather(
            *(self._update_claim(claim, evidence) for claim in selected),
            return_exceptions=True,
        )
        updates: list[dict[str, Any]] = []
        for claim, result in zip(selected, raw_updates):
            if isinstance(result, BaseException):
                print(
                    f"  claim update {claim['claim_id']} failed: {result}; retaining claim"
                )
                result = {
                    "claim_id": claim["claim_id"],
                    "original_claim": claim["claim"],
                    "is_included": False,
                    "should_update": False,
                    "updated_claim": claim["claim"],
                    "error": str(result),
                }
            updates.append(result)
        update_by_id = {item["claim_id"]: item for item in updates}

        final_claims: list[dict[str, Any]] = []
        selected_ids = {claim["claim_id"] for claim in selected}
        all_processed_claims: list[dict[str, Any]] = []
        for claim in claims:
            claim_id = claim["claim_id"]
            update = update_by_id.get(claim_id, {})
            included = bool(update.get("is_included", False))
            confidence = (
                1.0
                if claim_id in selected_ids and included
                else claim["uncertainty_metrics"][self.config.centrality_key]
            )
            final_text = (
                update.get("updated_claim", claim["claim"])
                if update.get("should_update", False)
                else claim["claim"]
            )
            processed = {
                **claim,
                "boundary_verifiable": boundary_by_id[claim_id]["verifiable"],
                "selected_for_verification": claim_id in selected_ids,
                "is_included": included,
                "was_updated": bool(update.get("should_update", False)),
                "final_claim": final_text,
                "post_verification_confidence": confidence,
            }
            all_processed_claims.append(processed)
            if confidence >= self.config.filtering_threshold:
                final_claims.append(processed)
        write_json(
            question_dir / "05_claim_updates.json",
            {
                "updates": updates,
                "filtering_threshold": self.config.filtering_threshold,
                "claims": all_processed_claims,
                "retained_claim_ids": [item["claim_id"] for item in final_claims],
            },
        )

        if final_claims:
            try:
                final_answer = await self.llm.complete(
                    final_answer_prompt(
                        question, [item["final_claim"] for item in final_claims]
                    ),
                    temperature=self.config.final_temperature,
                    max_tokens=1800,
                )
            except Exception as exc:  # noqa: BLE001 - retained claims are a valid fallback.
                print(
                    f"  final answer generation failed: {exc}; joining retained claims"
                )
                final_answer = " ".join(item["final_claim"] for item in final_claims)
        else:
            final_answer = "No claim met the configured final confidence threshold."

        result = {
            "question_index": question_index,
            "dataset_index": dataset_index,
            "question": question,
            "final_answer": final_answer,
            "final_claims": final_claims,
            "counts": {
                "initial_answers": len(answers),
                "merged_claims": len(claims),
                "boundary_verifiable": sum(item["verifiable"] for item in boundary),
                "selected_for_verification": len(selected),
                "updated": sum(item.get("should_update", False) for item in updates),
                "retained": len(final_claims),
            },
            "artifact_dir": str(question_dir.relative_to(self.run_dir)),
        }
        write_json(question_dir / "06_final_output.json", result)
        return result

    async def run(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        write_json(
            self.run_dir / "run_config.json",
            {
                "dataset": self.dataset.name,
                "dataset_path": str(self.dataset.path),
                "builtin": self.dataset.builtin,
                "provider": self.config.provider,
                "model": self.config.model,
                "num_answers": self.config.num_answers,
                "centrality": self.config.centrality,
                "centrality_key": self.config.centrality_key,
                "selection_threshold": self.config.selection_threshold,
                "filtering_threshold": self.config.filtering_threshold,
                "temperatures": {
                    "answer": self.config.answer_temperature,
                    "decomposition": self.config.decomposition_temperature,
                    "boundary": self.config.boundary_temperature,
                    "tool": self.config.tool_temperature,
                    "update": self.config.update_temperature,
                    "final": self.config.final_temperature,
                },
            },
        )
        results: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            try:
                results.append(await self.process_question(record, index))
            except Exception as exc:  # noqa: BLE001 - isolate failures by question.
                print(f"[{index}] pipeline failed: {exc}")
                error = {
                    "question_index": index,
                    "dataset_index": int(record.get("_dataset_index", index)),
                    "question": record["question"],
                    "error": str(exc),
                }
                question_dir = self.run_dir / "questions" / f"{index:04d}"
                write_json(question_dir / "error.json", error)
                results.append(error)
            write_json(self.run_dir / "06_final_output.json", results)
        return results
