# SimulRAG

Official implementation of [**SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA**](https://arxiv.org/abs/2509.25459).

SimulRAG first samples answers without retrieval, decomposes them into atomic claims, and estimates claim uncertainty from an answer-claim entailment graph. Simulator Boundary Assessment (SBA) identifies claims supported by an available scientific simulator. Low-confidence, simulator-verifiable claims are then selectively verified, updated, and assembled into a final answer.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set at least one provider key in `.env`. The default is OpenAI with `gpt-5-nano`.

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

## Quick start

The included Climate, Epidemiology, and Urban datasets replay the simulator evidence used during benchmark construction. They do not require a local simulator installation.

The release contains 1,000 Climate questions, 1,000 Epidemiology questions, and 200 Urban questions, matching the paper. Public records contain only `question` and `reference_answer`; the generation pipeline reads only `question`. References are loaded separately and only by `eval.py`.

```bash
python main.py --dataset climate --count 1
python eval.py --dataset climate
```

Built-in aliases include `climate`/`clim`, `epidemiology`/`epi`, and `urban`/`sumo`. `--count` defaults to one; use `--start` to select the first question index.

Paper defaults are `m=5`, closeness centrality, `tau=0.4`, `kappa=0.4`, answer temperature `1.0`, and decomposition/boundary/tool temperatures `0.1`. Every value is configurable:

```bash
python main.py --dataset epi --count 5 -m 5 --tau 0.4 --kappa 0.4 \
  --provider openai --model gpt-5-nano
```

The uncertainty score can be selected with `--centrality closeness`, `degree`, `pagerank`, `eigenvalue`, or `betweenness`. The selected metric is used consistently for both `tau` selection and `kappa` filtering. Centrality scales differ substantially, so thresholds should be calibrated separately when changing metrics.

Each run is written to `data/runs/<dataset>/<timestamp>/`. Per-question artifacts expose every stage:

```text
00_input.json
01_initial_answers.json
02_claims.json
03_boundary_analysis.json
04_simulator_output.json
05_claim_updates.json
06_final_output.json
07_evaluation.json
```

## Custom data and simulators

A custom dataset is a JSON array with exactly the fields used by the public benchmark files:

```json
[
  {
    "question": "An open scientific question",
    "reference_answer": "A reference answer used only by eval.py"
  }
]
```

Provide a simulator handbook and either a local executable/Python script or an HTTP endpoint:

```bash
python main.py \
  --dataset custom \
  --data examples/custom_questions.json \
  --handbook examples/custom_handbook.md \
  --simulator examples/custom_simulator.py
```

For a local simulator, SimulRAG sends one JSON object to stdin and expects JSON or text on stdout:

```json
{"tool": "function_name", "arguments": {"parameter": "value"}}
```

For an HTTP target, the same object is sent as a `POST` JSON body. The handbook must list exact function names, required parameters, types, units, valid ranges, and output fields. Invalid model output or failed simulator execution is retried with the execution error supplied as feedback.

## Creating a dataset

`src/create_data.py` converts simulator-derived question/answer pairs into open questions and reference answers. Its input is a JSON array containing `derived_quantitative_question` and `derived_quantitative_answer`; an optional `topic` gives generation context.

```bash
python -m src.create_data \
  --input raw/simulator_pairs.json \
  --output my_dataset/questions.json \
  --cache-output cache/my_dataset/simulator_evidence.json \
  --variations 2
```

The public output contains only `question` and `reference_answer`. The optional cache stores simulator evidence separately and can be adapted into a built-in replay dataset.

Existing benchmark JSON files that use `open_question` can be split without an LLM call. `--skip-incomplete` excludes records that cannot be replayed because required evidence is absent, and `--limit` makes the published sample count explicit:

```bash
python -m src.create_data \
  --normalize-existing \
  --skip-incomplete \
  --limit 1000 \
  --input raw/final_answers_climate.json \
  --output climate/questions.json \
  --cache-output cache/climate/simulator_evidence.json
```

For live generation, input items may omit the quantitative answer. Supply a handbook and simulator; the script first executes the same JSON tool-call interface used by `main.py`:

```bash
python -m src.create_data \
  --input raw/quantitative_questions.json \
  --output my_dataset/questions.json \
  --cache-output cache/my_dataset/simulator_evidence.json \
  --handbook examples/custom_handbook.md \
  --simulator examples/custom_simulator.py
```

A complete runnable source item is included at `examples/custom_quantitative_questions.json`.

## Project structure

```text
main.py                  # Pipeline CLI
eval.py                  # LLM evaluation CLI
data/                    # Public benchmarks, handbooks, caches, and ignored runs
examples/                # Minimal custom simulator integration
src/config.py            # Paper defaults
src/data_io.py           # Dataset and artifact I/O
src/llm_helper.py        # OpenAI, Anthropic, and Gemini routing
src/simulator.py         # Cached, subprocess, and HTTP adapters
src/pipeline.py          # SimulRAG stages
src/evaluation.py        # Final-answer and claim judges
src/create_data.py       # Open-question/reference generation
```

## Citation

If you use SimulRAG, please cite:

```bibtex
@article{xu2025simulrag,
  title={SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA},
  author={Xu, Haozhou and Wu, Dongxia and Chinazzi, Matteo and Niu, Ruijia and Yu, Rose and Ma, Yi-An},
  journal={arXiv preprint arXiv:2509.25459},
  year={2025}
}
```

## Installing the scientific simulators

These installations are optional for the bundled datasets because their simulator evidence is cached. They are useful for generating new scenarios or implementing a live adapter.

### Climate emulator

The climate emulator and checkpoint follow the Climate Tools setup from [Adapting While Learning](https://github.com/Rose-STL-Lab/Adapting-While-Learning):

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download Bohan22/AWL_Emulators --repo-type dataset --local-dir external/awl_emulators
mkdir -p simulators/climate
cp -R external/awl_emulators/climate_tool/. simulators/climate/
```

Install that repository's Python requirements before running the emulator. Its README also documents the device setting in `Climate_online/model/tas_reanalysis.yaml` and links the underlying MFRNP emulator.

### GLEAM-AI

The epidemiology interface follows [Rose-STL-Lab/GLEAM-API](https://github.com/Rose-STL-Lab/GLEAM-API):

```bash
git clone https://github.com/Rose-STL-Lab/GLEAM-API external/GLEAM-API
python -m pip install -r external/GLEAM-API/requirements.txt
fastapi run external/GLEAM-API/app/main.py --port 8000
```

The service requires the cloud/model configuration described by that repository. See the [GLEAM project](https://www.gleamproject.org/) for simulator background and data access details.

### SUMO

Use the official [Eclipse SUMO installation guide](https://eclipse.dev/sumo/docs/Installing/index.html). A compact Python installation is:

```bash
python -m pip install eclipse-sumo
python -c "import sumo; print(sumo.SUMO_HOME)"
```

For system packages, source builds, scenarios, and platform-specific instructions, follow the official SUMO documentation rather than pinning those dependencies in this library.

## License

SimulRAG is released under the [Apache License 2.0](LICENSE).
