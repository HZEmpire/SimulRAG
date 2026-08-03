from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.config import CENTRALITY_METRICS, PipelineConfig
from src.create_data import run as create_data_run
from src.data_io import (
    PROJECT_ROOT,
    Dataset,
    load_builtin_evidence,
    load_dataset,
)
from src.evaluation import evaluate_run
from src.pipeline import SimulRAGPipeline
from src.simulator import CachedSimulator, CustomSimulator


class FakeLLM:
    """Deterministic provider used to exercise orchestration without API access."""

    async def complete(
        self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 1200
    ) -> str:
        del temperature, max_tokens
        if "Please provide a comprehensive answer" in prompt:
            return (
                "The simulated intervention may change the target outcome. "
                "Decision makers should also consider implementation constraints."
            )
        if "AVAILABLE CLAIMS" in prompt:
            return "The simulator-grounded intervention changes the target outcome."
        raise AssertionError(f"Unexpected text prompt: {prompt[:80]}")

    async def complete_json(
        self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 1200
    ) -> Any:
        del temperature, max_tokens
        lowered = prompt.lower()
        if "deconstruct the following paragraph" in lowered:
            return [
                {
                    "claim": "The intervention changes the simulated target outcome.",
                    "confidence": 0.2,
                },
                {
                    "claim": "Decision makers should consider implementation constraints.",
                    "confidence": 0.8,
                },
            ]
        if "two sets of claims" in lowered:
            return [[0, 0], [1, 1]]
        if "judge every answer-claim pair" in lowered:
            return {"supported_claim_ids": [0, 1]}
        if "determine whether each claim" in lowered:
            verifiable = "implementation constraints" not in lowered
            return {
                "claims": [
                    {
                        "claim_id": 0,
                        "verifiable": verifiable,
                        "reason": "simulated output" if verifiable else "policy judgment",
                    }
                ]
            }
        if "exact top-level keys" in lowered:
            return {
                "tool": "compound_growth",
                "arguments": {
                    "initial_value": 100,
                    "rate_percent": 5,
                    "years": 3,
                },
            }
        if "fact-checking assistant" in lowered:
            return {
                "is_included": True,
                "should_update": True,
                "updated_claim": "The simulator confirms a change in the target outcome.",
            }
        if "evaluate a generated scientific answer" in lowered:
            return {"is_correct": True, "score": 0.9, "reason": "consistent"}
        if "evaluate whether a generated atomic claim" in lowered:
            return {"is_correct": True, "is_relevant": True, "reason": "supported"}
        if "create one realistic, open-ended scientific question" in lowered:
            return {
                "question": "What does the simulated growth imply?",
                "reference_answer": "The model shows positive compound growth.",
            }
        raise AssertionError(f"Unexpected JSON prompt: {prompt[:80]}")


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def config(self) -> PipelineConfig:
        return PipelineConfig(
            num_answers=2,
            selection_threshold=0.8,
            filtering_threshold=0.4,
            concurrency=4,
        )

    def test_supported_centrality_metrics(self) -> None:
        self.assertEqual(
            set(CENTRALITY_METRICS),
            {"closeness", "degree", "pagerank", "eigenvalue", "betweenness"},
        )
        for centrality, key in CENTRALITY_METRICS.items():
            config = PipelineConfig(centrality=centrality)
            config.validate()
            self.assertEqual(config.centrality_key, key)

    async def test_cached_climate_pipeline_and_evaluation(self) -> None:
        dataset = load_dataset("climate")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            pipeline = SimulRAGPipeline(
                dataset=dataset,
                config=self.config(),
                llm=FakeLLM(),
                simulator=CachedSimulator(load_builtin_evidence("climate")),
                handbook=(PROJECT_ROOT / "data/handbooks/climate.md").read_text(
                    encoding="utf-8"
                ),
                run_dir=run_dir,
            )
            results = await pipeline.run(dataset.records[:1])
            self.assertIn("final_answer", results[0])
            self.assertEqual(results[0]["counts"]["selected_for_verification"], 1)
            claims = json.loads(
                (run_dir / "questions/0000/02_claims.json").read_text()
            )["merged_claims"]
            self.assertTrue(claims)
            self.assertTrue(
                set(CENTRALITY_METRICS.values()).issubset(
                    claims[0]["uncertainty_metrics"]
                )
            )
            evidence = json.loads(
                (run_dir / "questions/0000/04_simulator_output.json").read_text()
            )
            self.assertEqual(evidence["source"], "built-in-cache")
            evaluated = await evaluate_run(run_dir, FakeLLM())
            self.assertTrue(evaluated[0]["answer_evaluation"]["is_correct"])
            self.assertEqual(evaluated[0]["claim_accuracy"], 1.0)

    async def test_custom_subprocess_simulator(self) -> None:
        dataset = load_dataset("custom", custom_path="examples/custom_questions.json")
        handbook = (PROJECT_ROOT / "examples/custom_handbook.md").read_text(
            encoding="utf-8"
        )
        simulator_path = str(PROJECT_ROOT / "examples/custom_simulator.py")
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = SimulRAGPipeline(
                dataset=Dataset(
                    name="custom",
                    path=dataset.path,
                    records=dataset.records,
                    builtin=False,
                ),
                config=self.config(),
                llm=FakeLLM(),
                simulator=CustomSimulator(simulator_path),
                handbook=handbook,
                run_dir=Path(temporary),
            )
            results = await pipeline.run(dataset.records[:1])
            self.assertIn("final_answer", results[0])
            evidence_path = Path(temporary) / "questions/0000/04_simulator_output.json"
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["source"], "custom-subprocess")
            self.assertAlmostEqual(
                evidence["simulator_output"]["final_value"], 115.7625
            )

    async def test_create_data_with_live_simulator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.json"
            output_path = root / "questions.json"
            cache_path = root / "evidence.json"
            raw_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "A value starts at 100 and grows 5 percent for 3 years."
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input=raw_path,
                output=output_path,
                cache_output=cache_path,
                variations=1,
                limit=None,
                normalize_existing=False,
                skip_incomplete=False,
                handbook=PROJECT_ROOT / "examples/custom_handbook.md",
                simulator=str(PROJECT_ROOT / "examples/custom_simulator.py"),
                provider="openai",
                model="gpt-5-nano",
                temperature=1.0,
                max_retries=3,
                concurrency=4,
            )
            with patch("src.create_data.LLMClient", return_value=FakeLLM()):
                await create_data_run(args)
            public = json.loads(output_path.read_text())
            evidence = json.loads(cache_path.read_text())
            self.assertEqual(set(public[0]), {"question", "reference_answer"})
            self.assertIn("final_value", evidence[0]["derived_quantitative_answer"])

    async def test_normalize_existing_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "legacy.json"
            output_path = root / "questions.json"
            cache_path = root / "evidence.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "open_question": "What changes in the scenario?",
                            "reference_answer": "The simulated value decreases.",
                            "derived_quantitative_answer": '{"change": -4.2}',
                            "final_model_answer": "unused",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input=input_path,
                output=output_path,
                cache_output=cache_path,
                variations=1,
                limit=None,
                normalize_existing=True,
                skip_incomplete=False,
            )
            await create_data_run(args)
            public = json.loads(output_path.read_text())
            evidence = json.loads(cache_path.read_text())
            self.assertEqual(set(public[0]), {"question", "reference_answer"})
            self.assertEqual(
                evidence[0]["derived_quantitative_question"], public[0]["question"]
            )


if __name__ == "__main__":
    unittest.main()
