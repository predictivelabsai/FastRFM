# FastRFM

Score one relational question three ways from the command line: a flattened recency/frequency/monetary model (`flat`), a local in-context model (`rfm`), and hosted KumoRFM (`kumo`).

`flat` and `rfm` run on this machine. They need no account. The `kumo` column calls Kumo's hosted model, and only that column needs an API key.

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

Lower MAE is better. `needs key` means `KUMO_API_KEY` is unset. The other two columns do not contact a network, and the command still exits 0.

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

## Get a Kumo API key

Check what this environment can see. The command prints `set` or `missing`. It does not print the key.

```bash
uv run fastrfm keys
```

Install the SDK, put the key in the environment, and score one task. Churn, lifetime value, and the restaurant question are one hosted call each. Fraud and the Formula 1 tasks batch by day, up to `--kumo-max-calls` (default 8). Start with a single call: the SDK uploads the tables.

```bash
uv sync --extra kumo
export KUMO_API_KEY="paste-the-key-here"
uv run fastrfm eval --tasks churn --kumo-max-calls 1
```

pip equivalent: `pip install 'fastrfm[kumo]'`, which installs `kumoai`.

Leave the key out of the repo, out of notebooks you commit, and out of shell history if you can avoid it. Setting the variable for one command works:

```bash
KUMO_API_KEY="paste-the-key-here" uv run fastrfm eval --tasks churn --kumo-max-calls 1
```

What the call does. `fastrfm` imports `kumoai` and calls `rfm.init()`, which reads `KUMO_API_KEY`. The graph is cut to rows before the anchor, so the latest timestamp Kumo sees is the evaluation cutoff. The question goes as a Predictive Query, for example:

```text
PREDICT COUNT(orders.*, 0, 90, days)=0 FOR EACH customers.customer_id
```

The weights stay on Kumo's side. They are not downloaded.

### Where the key comes from

`kumoai` 2.22, the SDK this extra installs, creates a KumoRFM key at <https://kumorfm.ai/api-keys>. Its own login helper opens <https://kumorfm.ai/authenticate-sdk/> and writes `KUMO_API_KEY` for you. The allowance published with that product was 1,000 queries a day.

On 2 October 2026 those URLs, including `https://kumorfm.ai/api`, redirect to the [NVIDIA Kumo Relational docs](https://docs.nvidia.com/sdgm/rfm/overview). There is no signup form on that page. A key you already hold only works if you also point the SDK at a host that still answers:

```bash
export KUMO_API_KEY="paste-the-key-here"
export RFM_API_URL="https://your-kumo-host/api"
uv run fastrfm eval --tasks churn --kumo-max-calls 1
```

`RFM_API_URL` defaults to `https://kumorfm.ai/api`. That is the variable `rfm.init()` reads. `KUMO_API_ENDPOINT` is a different setting, used by the training SDK when no URL is passed, and this CLI does not pass it through.

If your organization runs Kumo Studio, the Studio key is created there:

1. Open **API Keys**, or go to `https://<your-kumo-platform-dns>/api-keys`.
2. Choose **Create API key**, give it a name, and copy the value. Studio shows the full key only at creation. It has the shape `customer_id:secret`.
3. The steps are written up in [API Key Management](https://docs.nvidia.com/sdgm/fine-tuning/api-key-management).

That key authenticates Kumo Predict (`kumoai.init` against your studio). `fastrfm` sends `KUMO_API_KEY` to the RFM predict endpoint in `RFM_API_URL`. Set `RFM_API_URL` to a host that serves KumoRFM predict.

NVIDIA's current Kumo Relational product is a NIM plus the package `kumo-relational-client`. Its connect guide is [Deploy, Install, and Connect](https://docs.nvidia.com/sdgm/rfm/sdk-getting-started). A gateway key for that client is `KUMO_RELATIONAL_API_KEY`. `fastrfm` does not import that package, so a NIM key leaves the `kumo` column as `needs key`.

If the key is missing, or `kumoai` is not installed, eval prints the reason once and still scores `flat` and `rfm`.

## The three models

| Model | What it does | Key |
| --- | --- | --- |
| `flat` | Logistic or ridge regression on recency, frequency, and monetary value. Every training label. No joins. | None |
| `rfm` | Retrieves a few labeled neighborhoods and predicts in one pass. No task weights. | None |
| `kumo` | Hosted [KumoRFM](https://docs.nvidia.com/sdgm/rfm/overview) through `kumoai`. | `KUMO_API_KEY` |

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
  kumo.py         # hosted KumoRFM, skipped without a key
  relbench.py     # fetch and score rel-f1
  evaluate.py
  report.py
tests/
```

## License

MIT. See [LICENSE](LICENSE).

RelBench `rel-f1` is separate. It is CC BY-SA 4.0 from the [Stanford STAR project](https://github.com/stanford-star/relbench), fetched at runtime into `data/` (gitignored).
