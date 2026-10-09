# Experiment code

[Back to FunDM](../README.md).

Use `main.py` from the repo root to reproduce paper experiments. For
custom label-free training, start with `mcq.py extract`, `train`, and `predict`.

| File | Purpose |
|---|---|
| [mcq.py](mcq.py) | FunDM extraction, head fitting, prediction, CE controls, and overlap checks. |
| [heads.py](heads.py) | Base, +1T, and CLM-shaped heads; CE loss and joint prompt construction. |
| [data.py](data.py) | Dataset loaders, pinned revisions, and the default exit depth. |
| [baselines.py](baselines.py) | Released CLM evaluation and MCQ fine-tuning data preparation. |
| [coderforge.py](coderforge.py) | Agent runs, teacher judgments, FunDM features, baselines, and summaries. |
| [deepswe.py](deepswe.py) | Supervised separate-state heads on CLM's released DeepSWE embeddings. |
| [teacher_prompt_val.py](teacher_prompt_val.py) | Teacher prompt comparison on held-out training questions. |
| [extras.py](extras.py) | Full-model readout checks and the separate PyTorch timing experiment. |
| [prepare_serving_inputs.py](prepare_serving_inputs.py) | Matched native vLLM workloads and head/reference preparation. |
| [prepare_clm_serving_inputs.py](prepare_clm_serving_inputs.py) | The matching CLM workload and reference probabilities. |
| [fundm_vllm.py](fundm_vllm.py) | Native pooling adapter, teacher readout, and memory probes. |
| [serve_benchmark.py](serve_benchmark.py) | Teacher and four FunDM variants: memory, throughput, and latency. |
| [serve_clm_benchmark.py](serve_clm_benchmark.py) | Released CLM serving measurements with caching enabled. |

Direct commands also run from the repo root, for example:

```bash
.venv/bin/python scripts/mcq.py train --help
```

See the [reproduction guide](../docs/reproduction.md) for complete recipes.
