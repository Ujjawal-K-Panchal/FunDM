# Correctness checks

[Back to FunDM](../README.md).

| File | Covers |
|---|---|
| [test_label_free.py](test_label_free.py) | Label-independent fitting, optional evaluation, cache merging, and single-row raw/cached inference. |
| [test_reproduction.py](test_reproduction.py) | Automatic CoderForge holdout creation, reuse, and preparation with existing embedding caches. |
| [test_heads.py](test_heads.py) | Head scoring, padding isolation, variable option counts, permutation behavior, BF16 inputs, gradients, and the CLM scale cap. |
| [test_losses.py](test_losses.py) | KL and soft cross-entropy reference values, gradients, padding, target normalization, and row averaging. |

Run from the repo root after installing the main environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The checks use synthetic inputs and a small test model. They need no
downloaded model or datasets. Head and loss tests run on CPU and also
on CUDA when available.
