# Kumo Relational run 20261002T140012Z

Endpoint `POST https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions`, Authorization: Bearer <NVIDIA API key>. Synthetic FastRFM warehouse (seed 0), graph cut at 2024-06-24T00:00:00.

| task | kind | n | result | latency |
| --- | --- | ---: | --- | ---: |
| churn | binary | 400 | AUROC 0.879 (positive rate 0.37) | 15.43 s |
| ltv | regression | 400 | MAE 136.33 vs 144.33 for a constant (mean) guess | 40.84 s |
| notify | ranking | 252 | hit@1 0.6468 | 30.16 s |
| forecast | forecasting | 5 customers x 3 | MAE 27.59 (mean actual 73.25) | 2.12 s / customer |

## Forecast detail

|   customer_id |   step | window_start   |   predicted |   actual |
|--------------:|-------:|:---------------|------------:|---------:|
|           104 |      1 | 2024-06-24     |      105.49 |    58.39 |
|           104 |      2 | 2024-07-24     |      103.12 |   103.86 |
|           104 |      3 | 2024-08-23     |      100.5  |    61.12 |
|            15 |      1 | 2024-06-24     |      111.52 |    68.56 |
|            15 |      2 | 2024-07-24     |      110.86 |   100.57 |
|            15 |      3 | 2024-08-23     |      111.45 |    61.16 |
|            44 |      1 | 2024-06-24     |       57.86 |    49.43 |
|            44 |      2 | 2024-07-24     |      113.16 |    98.98 |
|            44 |      3 | 2024-08-23     |      111.47 |    61.28 |
|           313 |      1 | 2024-06-24     |      108.73 |    66.18 |
|           313 |      2 | 2024-07-24     |      102.79 |   102.75 |
|           313 |      3 | 2024-08-23     |      108.73 |    27.44 |
|           210 |      1 | 2024-06-24     |       82.15 |   105.9  |
|           210 |      2 | 2024-07-24     |       68.27 |    69.62 |
|           210 |      3 | 2024-08-23     |       64.71 |    63.47 |
