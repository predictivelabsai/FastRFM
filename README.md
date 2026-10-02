# FastRFM

Score one relational question three ways from the command line: a flattened recency/frequency/monetary model (`flat`), a local in-context model (`rfm`), and hosted NVIDIA Kumo Relational, formerly KumoRFM (`kumo`).

`flat` and `rfm` run on this machine. They need no account. The `kumo` column calls NVIDIA's hosted Kumo Relational model, and only that column needs an API key (`KUMO_API_KEY=nvapi-...` in `.env`, see below).

## Install

Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/predictivelabsai/FastRFM.git
cd FastRFM
uv sync
```

Or, with pip:

```bash
pip install -e ".[dev]"
```

## Run it

No key, no download. This builds a synthetic warehouse of customers, orders, payments, and support tickets, then prints the comparison.

```bash
uv run fastrfm eval
```

```text
task               metric        n       flat        rfm       kumo
churn              auroc       400      0.515      0.920  needs key
fraud              auroc      1260      0.526      0.955  needs key
ltv                mae         400    199.739    146.225  needs key
notify             hit@1       252      0.615      0.829  needs key
```

Lower MAE is better. `needs key` means `KUMO_API_KEY` is unset. With a key, see [Real results](#real-results-2-october-2026-hosted-nvidiakumo-relational). The other two columns do not contact a network, and the command still exits 0.

One customer the local models disagree on:

```bash
uv run fastrfm explain
```

Customer 72, test anchor 2024-06-24: last order 8 days ago, 7 orders, $224 in the past year. `flat` says P(churn)=0.07. Three severe account tickets and a 40% refund rate sit in the other tables. `rfm` says P(churn)=0.58, and the customer places no order in the next 90 days. The 0.58 is a distance-weighted average of the 31 nearest among 64 labeled customers. The AUROC is the ranking on the whole test split.

Other useful commands:

```bash
uv run fastrfm sources                         # datasets and tasks
uv run fastrfm generate -o data/warehouse      # write the parquet
uv run fastrfm eval --source data/warehouse --tasks churn,fraud
uv run fastrfm eval -o out/report.json         # keep the full report

# Stanford RelBench Formula 1. About a megabyte, CC BY-SA 4.0, no token.
uv run fastrfm fetch rel-f1
uv run fastrfm eval --source rel-f1

uv run pytest
```

`rfm` retrieves `--context-size` labeled rows (default 64) and averages `--k` neighbors (default 31). `flat` is fit on the whole training split.

## Hosted model: NVIDIA Kumo Relational (formerly KumoRFM)

KumoRFM moved under NVIDIA in 2026. kumorfm.ai no longer issues keys, and its old API (`https://kumorfm.ai/api`) now redirects to the [NVIDIA docs](https://docs.nvidia.com/sdgm/rfm/overview). The same model is now `nvidia/kumo-relational` on the NVIDIA API Catalog.

### Get a key

1. Sign in at <https://build.nvidia.com/nvidia/kumo-relational> (a free NVIDIA developer account is enough).
2. Click **Get API Key**. The key starts with `nvapi-`.
3. Put it in `.env` at the repo root. `.env` is gitignored, and `.env.example` shows the format:

```bash
cp .env.example .env      # then edit KUMO_API_KEY=nvapi-...
uv run fastrfm keys       # prints "set" or "missing" and the backend. Never prints the key.
```

`fastrfm` loads `.env` itself through python-dotenv. A real environment variable takes precedence. `NVIDIA_API_KEY` works too.

### How the call works

| | |
| --- | --- |
| Endpoint | `POST https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions` |
| Auth | `Authorization: Bearer $KUMO_API_KEY` |
| Request | Universal Structured Data request: task, schema, in-context labeled subgraphs, and the rows to predict |
| Client | `fastrfm.nvidia.KumoRelational(tables).predict(pql)` |

NVIDIA's `kumo-relational-client` / `kumo-relational-engine` 1.0.2 does the hard part. It takes a dict of DataFrames and a PQL query, then samples per-entity subgraphs and in-context examples and builds the request JSON. Its HTTP layer, though, targets a self-hosted NIM: it sends `X-API-Key` to `<url>/v1/predictions`. `fastrfm/nvidia.py` subclasses its `NimClient` to send `Authorization: Bearer` to `<catalog>/predictions` instead. Nothing else about the request changes. Sessions are disabled because the catalog is stateless, and `batch_mode("max")` splits large entity lists, for example 200 per call for ranking.

```python
from fastrfm.nvidia import KumoRelational
model = KumoRelational({"customers": customers, "orders": orders})   # pandas frames
pred = model.predict("PREDICT COUNT(orders.*, 0, 90, days)=0 FOR EACH customers.customer_id",
                     indices=[1, 2, 3], anchor_time=pd.Timestamp("2024-06-24"))
pred.frame      # ENTITY, ANCHOR_TIMESTAMP, PREDICTION, FALSE_PROB, TRUE_PROB
pred.seconds    # wall-clock latency of the hosted call(s)
```

The legacy `kumoai` SDK (`rfm.init(api_key=...)`, 2.22.0) does not work with an NVIDIA key. Its default host `kumorfm.ai/api` returns the NVIDIA docs HTML, and `init` fails with `JSONDecodeError`. `fastrfm` still falls back to `kumoai` for a key that does not start with `nvapi-`, in case you run your own Kumo host (`RFM_API_URL`).

### Run it

```bash
uv sync                                   # or: pip install -r requirements.txt && pip install -e .
uv run python scripts/kumo_demo.py        # 4 PQL tasks, writes results/<UTC timestamp>/
uv run fastrfm eval --kumo-max-calls 8    # flat vs rfm vs kumo on the synthetic warehouse
uv run fastrfm eval --source rel-f1 --kumo-max-calls 40   # RelBench F1, every test anchor
```

## Real results (2 October 2026, hosted `nvidia/kumo-relational`)

These are real predictions from the hosted NVIDIA API Catalog model with the free developer key. Raw outputs, per-entity CSVs, and JSON reports are in [`results/20261002T140011Z/`](results/20261002T140011Z/). Each graph was cut at the anchor, so the model saw no rows after it. Labels come from the rows after the anchor and are used only for scoring.

### `scripts/kumo_demo.py` on the synthetic warehouse (400 customers, anchor 2024-06-24)

| Task | PQL | n | Result | Latency |
| --- | --- | ---: | --- | ---: |
| Churn (binary) | `PREDICT COUNT(orders.*, 0, 90, days)=0 FOR EACH customers.customer_id` | 400 | **AUROC 0.879** (37% positive) | 15.4 s |
| 180-day spend (regression) | `PREDICT SUM(orders.net_amount, 0, 180, days) FOR EACH customers.customer_id` | 400 | **MAE 136.3**, against 144.3 for a constant guess. Mean predicted 261.0, mean actual 282.5 | 40.8 s |
| Next restaurant (ranking) | `PREDICT LIST_DISTINCT(orders.restaurant_id, 0, 30, days) RANK TOP 1 FOR EACH ...` | 252 | **hit@1 0.647** | 30.2 s (2 batches) |
| Spend forecast | `PREDICT SUM(orders.net_amount, 0, 30, days) FORECAST 3 TIMEFRAMES FOR customers.customer_id=<id>` | 5 x 3 | **MAE 27.6** per 30-day window (mean actual 73.3) | 2.1 s per customer |

Latency is wall clock from this machine and includes sampling the subgraphs locally. The same churn call took 15 s on one run and 41 s on another, so expect variance on the free tier.

### `fastrfm eval`: flat vs local rfm vs hosted Kumo Relational

Synthetic warehouse, test split:

| Task | Metric | n | flat | rfm | kumo |
| --- | --- | ---: | ---: | ---: | ---: |
| churn | AUROC | 400 | 0.515 | 0.920 | **0.879** |
| ltv | MAE | 400 | 199.7 | 136.7 | **136.3** |
| notify | hit@1 | 252 | 0.615 | 0.829 | **0.647** |
| fraud | AUROC | 1260 (kumo: 164 over 8 days) | 0.526 | 0.941 | **0.585** |

RelBench `rel-f1`, test split. Kumo scored every test row, with one call per anchor date (29–33 calls per task, 9 min total):

| Task | Metric | n | flat | rfm | kumo |
| --- | --- | ---: | ---: | ---: | ---: |
| driver-dnf | AUROC | 702 | 0.819 | 0.822 | **0.824** |
| driver-top3 | AUROC | 726 | 0.907 | 0.907 | **0.916** |
| driver-position | MAE | 760 | 4.039 | 3.000 | **2.683** |

What this shows:

- On real data (rel-f1), Kumo Relational is the best of the three on all three tasks with no training. The biggest gain is finishing position: MAE 2.68, against 3.00 for local rfm and 4.04 for flat.
- On the synthetic warehouse, the hand-built `rfm` features beat Kumo. That is expected, because the generator plants its signal in exactly the joins `features.py` aggregates. Kumo still clearly beats the RFM-only `flat` baseline on churn and ltv.
- Fraud is the weak spot. The signal sits two hops away (payment → shared device → other payments' labels), and only 164 test payments were scored, so 0.585 is a noisy estimate.
- The F1 queries draw harmless "semantic type" warnings, for example `results.statusId` inferred as an ID. Setting stypes explicitly on the graph would remove them.

## The three models

| Model | What it does | Key |
| --- | --- | --- |
| `flat` | Logistic or ridge regression on recency, frequency, and monetary value. Every training label. No joins. | None |
| `rfm` | Retrieves a few labeled neighborhoods and predicts in one pass. No task weights. | None |
| `kumo` | Hosted [NVIDIA Kumo Relational](https://build.nvidia.com/nvidia/kumo-relational) (ex-KumoRFM) through `fastrfm/nvidia.py`. | `KUMO_API_KEY` (NVIDIA `nvapi-` key) |

`rfm` has the shape of a relational foundation model: database in, question in, a handful of labeled subgraphs, an answer. The aggregates it retrieves over are written in `features.py`, so you can see which join moved the score. KumoRFM learns which aggregates to compute. [OpenRFM](https://arxiv.org/abs/2606.04320) and [RDB-PFN](https://github.com/MuLabPKU/RDBPFN) publish training recipes. This CLI does not call them.

## Synthetic warehouse

Six tables: `customers`, `restaurants`, `devices`, `orders`, `payments`, `tickets`.

Orders sit on a 12-day clock. Every 90 days a customer may pick up three severe account tickets and then place no orders for the next block. Earlier pauses are independent of this one, so recency, frequency, and spend have the same distribution for customers who pause and customers who do not. Fraud is planted on five shared devices: the cardholder's own history does not show it, and the other payments on that device do. A meal ticket makes the customer switch restaurant while their order history is still dominated by the old favorite.

The gaps in the table above are a property of this generator. They are a demonstration, not a RelBench score and not a measurement of KumoRFM-2.

## RelBench Formula 1

[rel-f1](https://huggingface.co/datasets/stanford-star/relbench-v1) is the smallest database in Stanford's RelBench: races, drivers, constructors, results, qualifying, standings. CC BY-SA 4.0. The Hub parquet includes the test labels, and `fastrfm eval --source rel-f1` scores that split. `flat` sees only that driver's own results. `rfm` also sees the constructor's other drivers, qualifying, and standings, all strictly before the anchor.

Same default of 64 context labels, test split:

| Task | Metric | n | flat | rfm |
| --- | --- | ---: | ---: | ---: |
| driver-dnf | AUROC | 702 | 0.819 | 0.822 |
| driver-top3 | AUROC | 726 | 0.907 | 0.907 |
| driver-position | MAE | 760 | 4.039 | 3.000 |

DNF and top-3 qualifying are already answered by the driver's own recent races, so the join barely moves AUROC. Mean finishing position improves once standings and the constructor's other drivers are visible. That is this database under this local model. The KumoRFM-2 paper's RelBench number is a different model on the full suite.

## Published KumoRFM-2 numbers

The figures in this section are published claims. The tables above are the local runs.

The write-up this tool is built from is TWIML AI Podcast episode 768, [Relational Foundation Models for Enterprise Data with Jure Leskovec](https://twimlai.com/podcast/twimlai/relational-foundation-models-enterprise-data) (21 May 2026; [YouTube](https://www.youtube.com/watch?v=khSSuUyvqno)). The paper linked from the show notes is [KumoRFM-2](https://arxiv.org/abs/2604.12596) (arXiv:2604.12596, 14 April 2026).

From that paper, in-context KumoRFM-2 uses at most 10,000 labeled examples and no per-task training:

- On the 12 RelBenchV1 binary tasks, its average AUROC is 79.60. RelGNN, the best supervised relational model in the same table, averages 78.06. The gap is 1.54 points. The introduction also reports about 5% over RelGNN on RelBenchV1 classification and regression, and about 10% over KumoRFM-1.
- On SAP SALT it reports about 8% over AutoGluon ensembles and about 25% over recent tabular foundation models. In the SALT table the in-context mean reciprocal rank is 0.83, against 0.77 for the data-scientist AutoGluon pipeline. Fine-tuning that table reaches 0.89: 6 points over the in-context model and 10 points over the best supervised baseline in the table. The introduction summarizes fine-tuning as about a 16% gain. On the podcast, Leskovec gave that gain as about 12% over the prior state of the art.
- Context can be as little as 0.2% of the training labels on the largest RelBench tasks. On rel-f1 the paper's own coverage is 83.5%.

Coinbase fraud, DoorDash restaurant recommendations, and the Reddit ad click-through lift are examples from the episode.

## Layout

```text
pyproject.toml
src/fastrfm/
  cli.py          # fastrfm keys | sources | generate | fetch | eval | explain
  warehouse.py    # synthetic customers, orders, payments, tickets
  tasks.py        # forward labels, cutoff at the anchor
  features.py     # aggregates with a strict ts < anchor cutoff
  models.py       # logistic / ridge, and neighborhood retrieval
  kumo.py         # PQL per task, cutoff, scoring; skipped without a key
  nvidia.py       # NVIDIA API Catalog client (Bearer auth) over kumo-relational-engine
  relbench.py     # fetch and score rel-f1
  evaluate.py
  report.py
tests/
scripts/kumo_demo.py  # real hosted predictions -> results/<timestamp>/
results/              # committed outputs of real runs
.env.example          # KUMO_API_KEY=nvapi-...
```

## License

MIT. See [LICENSE](LICENSE).

RelBench `rel-f1` is separate. It is CC BY-SA 4.0 from the [Stanford STAR project](https://github.com/stanford-star/relbench), fetched at runtime into `data/` (gitignored).
