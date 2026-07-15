# Qwen3-8B in-distribution preference prediction

## Run setup

- Tasks: 24 (`2 datasets x 4 negative ratios x 3 seeds`).
- Datasets: MovieLens 1M and Steam canonical `v1`.
- Split: random `0.7 / 0.1 / 0.2`; seeds `0, 1, 2`.
- Candidate construction: observed preference candidates only, cap 10 items, at most 1,000 evaluation users and 5 candidate groups per user.
- Targets: MovieLens rating at least 4/5; Steam total playtime at least 120 minutes.
- Main metric: test macro-by-user mean of candidate-group F1.
- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Serving: LiteLLM `1.92.0` with `simple-shuffle` routing over four vLLM `0.18.1` replicas, one A100 and tensor-parallel size 1 per replica.
- Generation: temperature 0, max 256 output tokens, maximum 20 history items, Qwen thinking disabled.
- Runtime: approximately 73 minutes from the first task start to the final metrics artifact.

## Results

Values are mean +/- sample standard deviation over three seeds. `Qwen strict F1` excludes failed candidate groups, while `failure-as-negative F1` retains them as incorrect/negative predictions. Popularity has complete coverage.

| Dataset | Ratio | Popularity F1 | Qwen strict F1 | Qwen failure-as-negative F1 | Qwen mean coverage | Failed rows / requested |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens 1M | 1 | 0.679413 +/- 0.006154 | 0.280401 +/- 0.006476 | 0.280401 +/- 0.006476 | 100.00% | 0 / 65,798 |
| MovieLens 1M | 2 | 0.538276 +/- 0.002573 | 0.230501 +/- 0.009995 | 0.230501 +/- 0.009995 | 100.00% | 0 / 64,317 |
| MovieLens 1M | 3 | 0.453241 +/- 0.004473 | 0.181705 +/- 0.006319 | 0.181705 +/- 0.006319 | 100.00% | 0 / 60,756 |
| MovieLens 1M | 9 | 0.255308 +/- 0.002677 | 0.093269 +/- 0.001556 | 0.093269 +/- 0.001556 | 100.00% | 0 / 61,540 |
| Steam | 1 | 0.718855 +/- 0.007037 | 0.071653 +/- 0.003103 | 0.069092 +/- 0.003275 | 96.06% | 1,362 / 34,592 |
| Steam | 2 | 0.594629 +/- 0.010659 | 0.066495 +/- 0.000998 | 0.063693 +/- 0.000323 | 94.02% | 2,541 / 42,453 |
| Steam | 3 | 0.520832 +/- 0.005326 | 0.046738 +/- 0.004430 | 0.045341 +/- 0.005098 | 95.42% | 2,144 / 46,800 |
| Steam | 9 | 0.320582 +/- 0.008425 | 0.023964 +/- 0.000276 | 0.022846 +/- 0.000349 | 95.92% | 2,150 / 52,690 |

## Coverage and caveats

- MovieLens: 28,756 candidate groups and 252,411 rows; no failed groups or rows.
- Steam: 21,694 candidate groups and 176,535 rows; 955 failed groups (4.40%) and 8,197 failed rows (4.64%).
- The Steam failures were strict-format failures, typically `Invalid yes/no answer for 'C1': ''`, after five deterministic attempts. The serving layer continued to return successful HTTP responses; no traceback, vLLM crash, CUDA OOM, or task-level failure occurred.
- The results show performance only for this target, split, candidate construction, prompt, and strict parser. They should not be described as model performance on MovieLens or Steam in general.

## Provenance and validation

- Base git commit: `002b89b0b7cab65d36af87df88f0e1aeac970fe7`.
- Source snapshot SHA-256: `22fd033a5cccf9379f8dfa72a01e2f11a058baca0f7e48fbefbd11a76c01d7ab`.
- Relevant source diff SHA-256: `f6568c5f6139c98d1f4207cc4e66eaf298facd2c1b081053892b12dc0fd2f517`.
- Copied run archive SHA-256: `d4d4cf67cc3fc79e79d217be69ef7e8448f80d596c416806a2b7f02d48daf2b5`.
- Artifact counts: 24 manifests, 24 pointwise metric files, 24 ranking metric files, 24 prediction parquet files, and 24 error JSONL files.
- Targeted tests: 49 passed in the experiment pod. Final local full suite: 331 passed with one pre-existing constant-input warning in the policy-ranking Spearman test.
