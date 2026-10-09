# CoderForge reproduction inputs

[Back to results](../README.md).

These files fix the runs and task partitions used by the experiments.

| File | Contents |
|---|---|
| [runs.tsv](runs.tsv) | Selected trajectory IDs and their `train`/`val` file assignment. |
| [data_sha256.txt](data_sha256.txt) | Checksums for the original run files; rebuilding checks `train.jsonl` and `val.jsonl`. |
| [tasks.json](tasks.json) | 200 training-pool tasks and 100 held-out test tasks. |
| `split_s0.json`–`split_s4.json` | Each seed's 180 training and 20 validation tasks, plus the same 100 test tasks. |

`main.py --table 3` rebuilds the CoderForge comparison. Its automatic
workflow exports `tasks.json`'s `holdout` list to
`data/clm_emb/coderforge-swerebench-8k/holdout.json` before CLM fine-tuning.

Raw trajectories and embeddings are generated under `data/`. Saved
FunDM depth/seed statistics are in
[agent-selection-data.json](../../assets/agent-selection-data.json).
