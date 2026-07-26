# T03 vs T04 — comparison on the curated 5-PDF set (LINKER_VERSION = 4)

Under the linker v4 fix (`_ART_RE_AZURE` accepts Markdown header prefixes) all
5 curated PDFs are precomputed under `LINKER_VERSION = 4` and evaluated. Doc 2
(`2004A27101.pdf`) is excluded because the BSARD GT vs source-PDF drift floors
every retriever at R@10 ≈ 0 on that PDF and gives no signal about the
retrievers themselves (its ground truth references a different Livre of the
same code).

Reproduce per-stem with:

```
python RQ2_T04_ARM2_METADATA/scripts/compare_t03_vs_t04.py --doc-id <stem> --smoke -v
```

Aggregate the 5 per-stem CSVs into the tables below with:

```
python RQ2_T04_ARM2_METADATA/scripts/aggregate_comparison.py
```

## Stems

| Stem | doc_id | Title | n_q |
|---|---:|---|---:|
| `1967_10_10_1967101056` | 5 | Code Judiciaire (smaller) | 71 |
| `1867_06_08_1867060850` | 6 | Code Pénal | 65 |
| `1804_03_21_1804032150` | 9 | Code Civil | 252 |
| `2003_07_17_2013A31614` | 8 | Loi (rentals) | 133 |
| `1967_10_10_1967101055` | 7 | Code Judiciaire (larger) | 204 |

`n_q` is the count of BSARD test+train questions whose ground truth
intersects the bsard_ids reachable in T04's v4 index for that PDF (each
question's GT is clipped to that intersection — "recoverable recall").

Analysis metrics: `R@10`, `R@100`, `MRR@10`, `nDCG@10`, plus `Cw/R@10` /
`Cw/R@100` where T07 cosine GT is populated (doc 9 only in this sweep).
`latency_ms` is captured in source CSVs for anomaly debug but is not a
reported metric.

## Aggregate metrics

### R@10

| method | doc 5 | doc 6 | doc 9 | doc 8 | doc 7 | mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm1_naive (T03)                | 0.3910 | **0.2985** | 0.4713 | 0.8006 | 0.3425 | 0.4608 | 0.4259 |
| arm2_metadata_node_raw          | **0.4678** | 0.2451 | 0.4735 | 0.7635 | 0.2935 | 0.4487 | 0.4582 |
| arm2_metadata_node_enriched     | **0.4678** | 0.2451 | 0.4735 | 0.7635 | 0.2935 | 0.4487 | 0.4582 |
| arm2_metadata_node_summary      | 0.3827 | 0.2395 | **0.5137** | **0.8045** | **0.3461** | 0.4573 | 0.4200 |
| arm2_metadata_node_full         | **0.4678** | 0.2451 | 0.4735 | 0.7635 | 0.2935 | 0.4487 | 0.4582 |
| arm2_metadata_article_raw       | 0.3964 | 0.2474 | 0.4766 | 0.7445 | 0.3091 | 0.4348 | 0.4156 |
| arm2_metadata_article_full      | 0.3964 | 0.2474 | 0.4766 | 0.7445 | 0.3091 | 0.4348 | 0.4156 |

### R@100

| method | doc 5 | doc 6 | doc 9 | doc 8 | doc 7 | mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm1_naive (T03)                | **0.9147** | 0.6399 | **0.9282** | **0.9499** | **0.7238** | 0.8313 | 0.8730 |
| arm2_metadata_node_raw          | 0.8608 | 0.5387 | 0.8520 | 0.8901 | 0.5947 | 0.7473 | 0.7996 |
| arm2_metadata_node_enriched     | 0.8608 | 0.5387 | 0.8520 | 0.8901 | 0.5947 | 0.7473 | 0.7996 |
| arm2_metadata_node_summary      | 0.8496 | **0.7059** | 0.8771 | 0.9371 | 0.6539 | 0.8047 | 0.8272 |
| arm2_metadata_node_full         | 0.8608 | 0.5387 | 0.8520 | 0.8901 | 0.5947 | 0.7473 | 0.7996 |
| arm2_metadata_article_raw       | 0.9025 | 0.6983 | 0.8930 | 0.9279 | 0.6341 | 0.8112 | 0.8521 |
| arm2_metadata_article_full      | 0.9025 | 0.6983 | 0.8930 | 0.9279 | 0.6341 | 0.8112 | 0.8521 |

### MRR@10

| method | doc 5 | doc 6 | doc 9 | doc 8 | doc 7 | mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm1_naive (T03)                | 0.2369 | 0.2836 | 0.3509 | 0.4605 | 0.3294 | 0.3323 | 0.3308 |
| arm2_metadata_node_raw          | **0.3371** | **0.3378** | 0.4397 | 0.6078 | 0.3922 | 0.4229 | 0.4076 |
| arm2_metadata_node_enriched     | **0.3371** | **0.3378** | 0.4397 | 0.6078 | 0.3922 | 0.4229 | 0.4076 |
| arm2_metadata_node_summary      | 0.3090 | 0.3362 | **0.4698** | **0.6262** | **0.4317** | 0.4346 | 0.4331 |
| arm2_metadata_node_full         | **0.3371** | **0.3378** | 0.4397 | 0.6078 | 0.3922 | 0.4229 | 0.4076 |
| arm2_metadata_article_raw       | 0.3122 | 0.3053 | 0.4480 | 0.5628 | 0.4051 | 0.4067 | 0.4059 |
| arm2_metadata_article_full      | 0.3122 | 0.3053 | 0.4480 | 0.5628 | 0.4051 | 0.4067 | 0.4059 |

### nDCG@10

| method | doc 5 | doc 6 | doc 9 | doc 8 | doc 7 | mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm1_naive (T03)                | 0.3054 | 0.3169 | 0.4182 | 0.5666 | 0.3937 | 0.4002 | 0.3969 |
| arm2_metadata_node_raw          | **0.3980** | **0.3626** | 0.4918 | 0.6643 | 0.4309 | 0.4695 | 0.4502 |
| arm2_metadata_node_enriched     | **0.3980** | **0.3626** | 0.4918 | 0.6643 | 0.4309 | 0.4695 | 0.4502 |
| arm2_metadata_node_summary      | 0.3633 | 0.3443 | **0.5135** | **0.6879** | **0.4780** | 0.4774 | 0.4777 |
| arm2_metadata_node_full         | **0.3980** | **0.3626** | 0.4918 | 0.6643 | 0.4309 | 0.4695 | 0.4502 |
| arm2_metadata_article_raw       | 0.3709 | 0.3344 | 0.4999 | 0.6348 | 0.4417 | 0.4563 | 0.4490 |
| arm2_metadata_article_full      | 0.3709 | 0.3344 | 0.4999 | 0.6348 | 0.4417 | 0.4563 | 0.4490 |

Bold = per-stem best across all 7 methods. Source: per-stem CSVs at
`../data/comparison_t03_vs_t04_<stem>.csv`, aggregated by
[`scripts/aggregate_comparison.py`](../scripts/aggregate_comparison.py).

### Cosine-weighted recall (doc 9 only — T07 GT not populated for the others)

| method | Cw/R@10 (doc 9) | Cw/R@100 (doc 9) |
|---|---:|---:|
| arm1_naive (T03)                | 0.4694 | **0.9287** |
| arm2_metadata_node_raw          | 0.4726 | 0.8504 |
| arm2_metadata_node_enriched     | 0.4726 | 0.8504 |
| arm2_metadata_node_summary      | **0.5149** | 0.8721 |
| arm2_metadata_node_full         | 0.4726 | 0.8504 |
| arm2_metadata_article_raw       | 0.4742 | 0.8976 |
| arm2_metadata_article_full      | 0.4742 | 0.8976 |

On doc 9 the cosine-weighted direction matches binary: `node_summary` leads
at K=10, T03 leads at K=100. No verdict flips from binary → cosine.

### Best T04 variant overall (lowest mean rank across stem × metric)

| method | mean_rank | n_cells |
|---|---:|---:|
| `arm2_metadata_node_summary`    | 2.591 | 22 |
| `arm2_metadata_article_full`    | 2.773 | 22 |
| `arm2_metadata_article_raw`     | 2.773 | 22 |
| `arm2_metadata_node_enriched`   | 2.955 | 22 |
| `arm2_metadata_node_full`       | 2.955 | 22 |
| `arm2_metadata_node_raw`        | 2.955 | 22 |

`node_summary` wins overall. The three node `raw / enriched / full` variants
land on identical numbers on every recall-type metric because they collapse
to the same token stream on a single-doc corpus once the constant
`[Document]` header is dropped; their rank ties downstream are expected, not a
bug.

## Interpretation (≤200 words)

**Verdict.** Under v4, T04 (best variant) beats T03 on R@10 on 4 of 5 stems,
and on MRR@10 + nDCG@10 on 5 of 5. T03 still owns R@100 on 4 of 5 stems —
a broader-net-vs-better-top-10 tradeoff that holds across the curated set.
`arm2_metadata_node_summary` is the strongest T04
variant overall (mean rank 2.591), and is the variant that wins MRR@10 and
nDCG@10 on doc 9, doc 8, and doc 7. The article-level variants come second.

**The one exception.** Doc 6 (Code Pénal) is the lone stem where T03 wins
R@10 (0.299 vs 0.247) — a deficit the linker v4 fix did not move. This is a
real T04 weakness on this PDF, not a linker artefact.

**On the doc-2 exclusion.** Doc 2 (`2004A27101.pdf`) is excluded because every
retriever floors at R@10 = 0 there, owing to an upstream BSARD-GT-vs-PDF drift
problem (different Livres of the same code sharing article numbers). That
problem is doc-2-specific; on the 5 curated PDFs no such drift is detected and
the numbers above are meaningful retrieval signal.

## Caveats

1. **Linker v4 as the methodological prerequisite.** The shift on doc 8 (T04
   R@10 0.05–0.22 → 0.74–0.80; MRR@10 0.04–0.08 → 0.56–0.63 from v3 → v4) is
   driven by the `_ART_RE_AZURE` fix that lets `Art.` markers be recognised
   when emitted with Markdown header prefixes (`##### Art. 219.`). Without that
   fix, doc 8 would have been the dominant outlier and the T04-vs-T03 verdict
   on R@10 would still show T03 winning. See the side-by-side at
   [`v3_vs_v4_doc8_retro.md`](v3_vs_v4_doc8_retro.md).
2. **n_q varies across stems** (65–252). Cross-stem `mean` rows treat each
   stem equally; weighting by `n_q` would shift mean values somewhat but
   not the per-stem winner column.
3. **Doc 6 R@10 deficit.** Per-question error analysis is reported in
   [`error_analysis_doc6_v4.md`](error_analysis_doc6_v4.md).
4. **Cw/R metrics only on doc 9.** Other stems lack T07 cosine GT in this
   sweep; treat as missing, not as zero.

## What ships

- [`comparison_summary_v4.md`](comparison_summary_v4.md) — headline + auto detail tables (this writeup's data source).
- [`comparison_summary_v4.csv`](comparison_summary_v4.csv) — machine-readable wide form (re-plot friendly).
- [`comparison_all_stems_v4.csv`](comparison_all_stems_v4.csv) — long form (one row per stem × method × metric).
- [`../data/comparison_t03_vs_t04_<stem>.csv`](../data/) — 5 per-stem source CSVs (T03 + 6 T04 SMOKE_PLAN variants).
- [`../data/comparison_per_query_<stem>.json`](../data/) — 5 per-stem per-query metrics + GT + cosine GT.
- Per-variant ranked-items JSONL at `../data/<stem>/results/<v4-hash>/compare_*.jsonl`.
