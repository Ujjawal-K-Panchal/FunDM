# Third-party dependencies

[Back to FunDM](../README.md).

[CLM/](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/README.md) is the pinned upstream Contrastive Language Model
repository used for our baselines. Its documentation and
[license](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/LICENSE) accompany the code.

| CLM location | Purpose |
|---|---|
| [src/clm/](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/) | Released scoring heads, decision schemas, embedding client, and caches. |
| [train/finetune.py](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/finetune.py) | The authors' fine-tuning implementation. |
| [evaluation/bon_eval.py](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/evaluation/bon_eval.py) | Agent-run selection evaluation. |
| [preprocessing/hf_embeddings.py](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/preprocessing/hf_embeddings.py) | Released embedding download and conversion. |
| [docs/FINETUNING.md](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/docs/FINETUNING.md) | Upstream training documentation. |

If the submodule is missing, run from the FunDM repo root:

```bash
git submodule update --init --recursive
```

FunDM-specific wrappers and comparisons are in
[scripts/](../scripts/README.md); their setup is in the
[reproduction guide](../docs/reproduction.md).
