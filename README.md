# OptiCom

**A Unified Framework for State-Conditioned Composition in LLM-Driven Optimization**

[Paper](https://arxiv.org/abs/2609.37221) · [Setup](#getting-started) · [Benchmarks](benchmarks/README.md) · [Results](#results) · [Citation](#citation)

OptiCom coordinates LLM-driven optimization through a shared configuration space,
`C = (A, Q, O, E, M, S)`: artifact, query, operator, evaluation, memory, and strategy.
A fast Optimization Controller composes actions using the current search state;
a slower Strategy Adapter updates preferences using accumulated feedback.

The paper reports an average **Max-score rank of 1.72** across **32 benchmark
groups**, with the top score in **23 groups**, among 14 evaluated configurations.
The 13 comparison profiles are method-inspired implementations within this
framework; the paper reports official-code comparisons separately.

## Framework

| Component | Responsibility | Implementation |
| --- | --- | --- |
| **A — Artifact** | Synthesize candidates and retain the incumbent | [artifact_synthesizer.py](src/artifact_synthesizer.py), [archive_manager.py](src/archive_manager.py) |
| **Q — Query** | Retrieve evidence and assemble context | [query_engine.py](src/query_engine.py), [context_assembler.py](src/context_assembler.py) |
| **O — Operator** | Select transformations such as revision, repair, and recombination | [operator_library.py](src/operator_library.py) |
| **E — Evaluation** | Execute and evaluate candidates | [execution_harness.py](src/execution_harness.py), [evaluation_stack.py](src/evaluation_stack.py) |
| **M — Memory** | Retain events and summarize experience | [optimization_memory.py](src/optimization_memory.py) |
| **S — Strategy** | Compose actions and adapt search preferences | [optimization_controller.py](src/optimization_controller.py), [strategy_adapter.py](src/strategy_adapter.py) |

The controller emits an Action Package containing `query_plan`, `context_scope`,
`operator_choice`, `branch_width`, `evaluation_depth`, and `archive_policy`.
These fields coordinate evidence gathering, candidate generation, evaluation,
and archive updates. Strategy adaptation uses accumulated trajectory feedback
to update controller preferences and operators.

[baseline_profiles.py](src/baseline_profiles.py) defines profiles inspired by
TextGrad, OPRO, ProTeGi, Reflexion, GEPA, AdaEvolve, EvoX, FunSearch, AlphaEvolve,
Voyager, DSPy, Self-Refine, and LATS. Use `baseline: opticom` for the full framework.

## Repository layout

```text
OptiCom/
├── README.md
├── requirements.txt
├── run.py                      # Single optimization run
├── run_baselines.py            # Compare baseline profiles
├── run_all_benchmarks.sh        # Benchmark sweep
├── run_baseline_per_task.sh     # Baseline sweep for one task
├── summarize_results.py        # Tabulate archived scores
├── summarize_runtimes.py       # Tabulate archived runtime records
├── rescore_results.py          # Re-evaluate archived artifacts
├── src/                        # Core framework
├── benchmarks/                 # Tasks, evaluators, and task-specific setup
├── example/                    # YAML configuration templates
├── result/                     # Released best-result archive
├── paths/                      # Supporting search artifacts
├── tests/                      # Framework and evaluator tests
└── docs/release-history/        # Original supplementary-package notes
```

## Getting started

Run the commands below from the repository root. The shell examples use Bash;
the sweep scripts require Bash. Docker, a GPU, compilers, datasets, or external
services may also be required by individual benchmarks.

### 1. Install dependencies

Use Python 3.10 or later, subject to the requirements of your selected dependencies.
The requirements file includes framework, benchmark, and development dependencies;
it is not a minimal installation or a fully pinned environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.
Install task-specific dependencies where applicable, for example:

```bash
python -m pip install -r benchmarks/math/circle_packing/requirements.txt
```

### 2. Configure the model

The example files are templates: **set `llm.model` to a model or endpoint ID
available to your account before running optimization**. Released templates use
`model: null`. Edit the provider and optional base URL for your service, and keep
`llm.api_key: null` so the client reads its key from the environment. Configurations
are copied into run outputs, so credentials should stay out of YAML files.

| Provider value | Environment variable |
| --- | --- |
| `openai`, `gpt` | `OPENAI_API_KEY` |
| `claude`, `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini`, `google` | `GOOGLE_API_KEY` |
| `qwen`, `dashscope` | `DASHSCOPE_API_KEY` |
| `minimax` | `MINIMAX_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |

The `mock` provider does not require a key and is intended for development;
it does not reproduce model-based experimental results.

| Template | Purpose |
| --- | --- |
| [benchmark_openai.yaml](example/benchmark_openai.yaml) | Benchmark-bound OpenAI configuration; examples below override the task |
| [benchmark_arc.yaml](example/benchmark_arc.yaml) | Alternative benchmark configuration |
| [circle_packing.yaml](example/circle_packing.yaml) | Standalone circle-packing prompt with benchmark binding disabled |

Relative `benchmark.root` paths are resolved from the configuration file's location.
If you copy a template elsewhere, update that path accordingly.

### 3. Discover tasks and run optimization

```bash
python run.py -c example/benchmark_openai.yaml --list-benchmarks
python run.py --list-baselines

# Run after configuring your model and the selected evaluator.
python run.py -c example/benchmark_openai.yaml \
    --benchmark circle_packing --max-iter 20 --no-warm-start
```

By default, the runner loads a previous best artifact from the matching result
directory. Use `--no-warm-start` for a fresh search. This flag disables loading;
it does not disable saving a new best result. Use a separate working copy when
running experiments if you want to preserve the released archive.

### 4. Compare profiles or sweep tasks

```bash
python run_baselines.py -c example/benchmark_openai.yaml \
    --benchmark circle_packing \
    --baselines textgrad,opro,reflexion,gepa --max-iter 5

bash run_all_benchmarks.sh -c example/benchmark_openai.yaml -i 20
bash run_baseline_per_task.sh -c example/benchmark_openai.yaml -b circle_packing
```

The sweep scripts install task requirements automatically unless passed
`--no-auto-install`. Check each task's setup before starting a broad sweep.

## Benchmarks

The repository contains task families spanning mathematical constructions,
systems optimization, GPU kernels, algorithmic contests, visual reasoning,
prompt optimization, image generation, and quantum circuits. Available task
instances depend on the family, external assets, and resolver configuration;
the paper's 32 benchmark groups are the experimental evaluation set.

| Family | Directory |
| --- | --- |
| Mathematical optimization | [math](benchmarks/math/) |
| Systems optimization | [ADRS](benchmarks/ADRS/) |
| GPU kernels | [gpu_mode](benchmarks/gpu_mode/), [kernelbench](benchmarks/kernelbench/) |
| Algorithmic contests | [frontier-cs-eval](benchmarks/frontier-cs-eval/), [ale_bench](benchmarks/ale_bench/) |
| Visual reasoning | [arc_benchmark](benchmarks/arc_benchmark/) |
| Prompt optimization | [prompt_optimization](benchmarks/prompt_optimization/) |
| Image generation | [image_gen](benchmarks/image_gen/) |
| Quantum circuits | [qnn_circuit_topology](benchmarks/qnn_circuit_topology/) |

See the [benchmark guide](benchmarks/README.md) for task layouts and evaluator
interfaces. Some family guides also document upstream tools; the commands above
are the entry points for this OptiCom package.

## Results

Released records are stored under:

```text
result/<baseline>/<task>/<model>/
├── best_program.*              # Extension depends on the artifact type
├── config.yaml
├── config.source.yaml          # Original configuration, when available
└── result.json
```

The archive contains `redacted-model` directory names from the supplementary
release. Summarize all archived records without a model filter:

```bash
python summarize_results.py
python summarize_results.py --pivot by_baseline_model --pivot-metric runtime_score
python summarize_runtimes.py --source result
```

Use `--baseline opticom` or `--task math__circle_packing` to narrow the score
summary. Passing `-c` to `summarize_results.py` filters by that configuration's
model and may exclude the anonymized archive.

These commands tabulate stored records; they do not rerun experiments or
independently reproduce the paper's aggregate rankings. See the
[paper](https://arxiv.org/abs/2609.37221) for evaluation protocols and aggregation.
The original [package manifest](docs/release-history/PACKAGE_MANIFEST.md) and
[v5 notes](docs/release-history/PACKAGE_NOTES_v5.md) are retained as historical
records; their snapshot-specific values may differ from later archived records.

## Tests

After installing the dependencies, run:

```bash
python -m pytest tests/ -q
```

This is the test entry point, not a claim that all benchmark families have been
validated on every platform. Full benchmark execution needs the corresponding
task dependencies and resources.

## Citation

```bibtex
@article{wei2026opticom,
  title         = {{OptiCom}: A Unified Framework for State-Conditioned Composition in LLM-Driven Optimization},
  author        = {Wei, Chenxing and Liu, Sichen and Liu, Lizhao and Sun, Ningyuan and Chen Bingzhou and He, Ying and Jiang, Bo and Yu, Fei and Shu, Yao},
  year          = {2026},
  eprint        = {2609.37221},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.37221}
}
```
