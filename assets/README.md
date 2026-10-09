# Figures and saved measurements

[Back to FunDM](../README.md).

This folder holds the paper's figures and archived experiment reports.

| File | Contents |
|---|---|
| [figure1.png](figure1.png) | CLM, the full-model teacher, and FunDM training/inference. |
| [figure2.png](figure2.png) | Supervised FunDM-CE and FunDM-CE+1T architectures. |
| [figure3.png](figure3.png) | Per-benchmark accuracy and teacher retention across exit depths. |
| [figure4.png](figure4.png) | CoderForge selection and AUC across exit depths, with seed SDs. |
| [exit-depth-data.json](exit-depth-data.json) | Numeric data behind Figures 3 and 4. |
| [agent-selection-data.json](agent-selection-data.json) | CoderForge depth/seed results and layer-24 paired teacher comparisons. |
| [serving-data.json](serving-data.json) | Native vLLM workload, validation, memory, throughput, and latency. |
| [timing-data.json](timing-data.json) | Separate PyTorch throughput measurements, including prepared CLM options. |
| [timing-provenance.json](timing-provenance.json) | PyTorch timing protocol, source hashes, and cache conditions. |

PyTorch and vLLM reports describe separate experiments. See the
[measurement guide](../docs/reproduction.md#inference-measurements)
for their settings and reproduction commands.
