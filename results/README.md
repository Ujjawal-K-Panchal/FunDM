# Experiment outputs

[Back to FunDM](../README.md).

[coderforge/](coderforge/README.md) contains the versioned run lists and
task splits. Other experiment outputs are generated locally and ignored
by Git. Archived release measurements live in [assets/](../assets/README.md).

| Generated location | Contents |
|---|---|
| `kd/`, `kd_target/`, `kd_big/` | MCQ FunDM results, target ablations, and expanded-pool results. |
| `head_*.json`, `comparator_sets/`, `grid/` | Original CE evaluations and encoding/head ablations. |
| `clm_choice/`, `clm_shaped_sets/` | Fine-tuned CLM and from-scratch MCQ controls. |
| `kd_cf/` | CoderForge FunDM metrics, predictions, checkpoints, and summary. |
| `joint_agentonly/`, `joint_agentonly_a/` | Action-trained CLM-shaped and CE controls. |
| `clm_ft_agent/`, `released_clm_agent/` | CLM agent fine-tuning, evaluations, and ADP overlap checks. |
| `deepswe_steps/` | Supervised heads on released DeepSWE embeddings. |
| `serving/` | Output directory used by the guide's native vLLM benchmark commands. |

Start with `main.py --help` from the repo root. The
[reproduction guide](../docs/reproduction.md) maps experiments to paper tables.
