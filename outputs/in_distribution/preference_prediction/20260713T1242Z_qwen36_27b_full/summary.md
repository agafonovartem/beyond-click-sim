# Qwen3.6-27B in-distribution preference prediction

## Run setup

- Tasks: 24 (`2 datasets x 4 negative ratios x 3 seeds`).
- Datasets: MovieLens 1M and Steam canonical `v1`.
- Split: random `0.7 / 0.1 / 0.2`; seeds `0, 1, 2`.
- Candidate construction: observed preference candidates only, cap 10 items, at most 1,000 evaluation users and 5 candidate groups per user.
- Targets: MovieLens rating at least 4/5; Steam total playtime at least 120 minutes.
- Main metric: test macro-by-user mean of candidate-group F1.
- Model: `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Serving: LiteLLM `1.92.0` over four vLLM `0.18.1` replicas, one A100 and tensor-parallel size 1 per replica.
- Generation: temperature 0, max 256 output tokens, maximum 20 history items, Qwen thinking disabled.
- Runtime: approximately three hours.

## Results

Values are mean +/- sample standard deviation over three seeds. Popularity and Qwen3-8B are the matching runs under the same tasks.

| Dataset | Ratio | Popularity F1 | Qwen3-8B F1 | Qwen3.6-27B F1 |
|---|---:|---:|---:|---:|
| MovieLens 1M | 1 | 0.679413 +/- 0.006154 | 0.280401 +/- 0.006476 | 0.595869 +/- 0.001003 |
| MovieLens 1M | 2 | 0.538276 +/- 0.002573 | 0.230501 +/- 0.009995 | 0.491398 +/- 0.003695 |
| MovieLens 1M | 3 | 0.453241 +/- 0.004473 | 0.181705 +/- 0.006319 | 0.415506 +/- 0.002372 |
| MovieLens 1M | 9 | 0.255308 +/- 0.002677 | 0.093269 +/- 0.001556 | 0.227566 +/- 0.005996 |
| Steam | 1 | 0.718855 +/- 0.007037 | 0.071653 +/- 0.003103 | 0.631637 +/- 0.005006 |
| Steam | 2 | 0.594629 +/- 0.010659 | 0.066495 +/- 0.000998 | 0.549780 +/- 0.005868 |
| Steam | 3 | 0.520832 +/- 0.005326 | 0.046738 +/- 0.004430 | 0.483103 +/- 0.007694 |
| Steam | 9 | 0.320582 +/- 0.008425 | 0.023964 +/- 0.000276 | 0.288638 +/- 0.002519 |

## Coverage and interpretation

- Qwen3.6-27B scored 428,946/428,946 requested rows: 100% coverage, zero failed rows, and zero failed candidate groups.
- Qwen3.6-27B is substantially stronger than Qwen3-8B in every dataset/ratio slice and eliminates the 8B model's Steam format failures.
- Popularity remains stronger in every slice under this specific target, split, candidate construction, prompt, and metric.
- These results do not include the Agent4Rec traits/persona pipeline; they evaluate the direct preference-specific yes/no prompt.

## Provenance and validation

- Base git commit: `002b89b0b7cab65d36af87df88f0e1aeac970fe7`.
- Source snapshot SHA-256: `79977d0aae492bee2ac898011991437eaa0c9416c14384fbf3c909aabd34991c`.
- Relevant source diff SHA-256: `bc1e515b30509f3ca1da4dd56b20ff15c9d3e0e7879290d384ad2e7ea7c6566f`.
- Copied run archive SHA-256: `d0f867be46b3ab8c5adbf51fb7b5cb21f2b3a0e254e1ee9070d2aef8945dc39f`.
- Artifact counts: 24 manifests, 24 pointwise metric files, 24 ranking metric files, 24 prediction parquet files, and 24 error JSONL files.
- Tests: 51 targeted tests passed in the pod; final local suite passed 333 tests with one constant-input warning in the policy-ranking Spearman test.
