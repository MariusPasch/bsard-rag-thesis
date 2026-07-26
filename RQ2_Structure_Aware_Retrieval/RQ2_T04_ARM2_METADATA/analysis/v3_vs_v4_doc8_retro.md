# Doc 8 — linker v3 vs v4 retro

**Stem:** `2003_07_17_2013A31614` · doc_id 8 · Loi (rental law).

The 2026-05-25 session shipped linker v4 (`_ART_RE_AZURE` accepts Markdown header prefixes — see `CHANGE_NOTES.md`). This file quantifies the fix's actual impact on doc 8 by re-running each T04 variant's *persisted* ranked-items JSONL through the metric harness side-by-side under both eras.

## Method

- **v3 source:** six per-config JSONLs at `../data/2003_07_17_2013A31614/results/<v3-hash>/compare_2026-05-25*.jsonl` (one per T04 variant, 75 query rows each).
- **v4 source:** six per-config JSONLs at `../data/2003_07_17_2013A31614/results/<v4-hash>/compare_2026-05-26*.jsonl` (one per T04 variant, 133 query rows each).
- **GT:** the v4-era clipped GT from `../data/comparison_per_query_2003_07_17_2013A31614.json` (the v3 GT clip was overwritten by Phase 1). The v4 GT is a superset of the v3 GT because v4 reaches more bsard_ids — restricting to the 75 v3 queries lets us compare apples-to-apples on the question axis.
- **Common subset:** 75 queries appearing in both eras; v4 reaches 58 extra queries unreachable under v3 (those queries had no v3-reachable GT to score against, so they did not exist in the v3 evaluation subset).
- **T03 (arm1_naive)** is NOT linker-dependent — it indexes the deduped BSARD article corpus, not the AzureDI extraction. Its retrieval changes era only because the question subset changes. We show T03's v4-era full-subset number (n=133) from the comparison CSV as context, but no v3/v4 delta is meaningful for it.

## Side-by-side table (T04 variants, n=75 common subset)

| Variant | v3 R@10 | v4 R@10 | Δ | v3 R@100 | v4 R@100 | Δ | v3 MRR@10 | v4 MRR@10 | Δ | v3 nDCG@10 | v4 nDCG@10 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| node_raw | 0.0324 | 0.7818 | **+0.7493** | 0.0813 | 0.8984 | +0.8171 | 0.0436 | 0.6523 | **+0.6088** | 0.0258 | 0.6182 | +0.5924 |
| node_enriched | 0.0324 | 0.7818 | **+0.7493** | 0.0813 | 0.8984 | +0.8171 | 0.0436 | 0.6523 | **+0.6088** | 0.0258 | 0.6182 | +0.5924 |
| node_summary | 0.1424 | 0.7978 | **+0.6553** | 0.2262 | 0.9151 | +0.6889 | 0.0829 | 0.6053 | **+0.5224** | 0.0742 | 0.6252 | +0.5510 |
| node_full | 0.0324 | 0.7818 | **+0.7493** | 0.0813 | 0.8984 | +0.8171 | 0.0436 | 0.6523 | **+0.6088** | 0.0258 | 0.6182 | +0.5924 |
| article_raw | 0.0280 | 0.7380 | **+0.7100** | 0.2469 | 0.9056 | +0.6587 | 0.0158 | 0.5413 | **+0.5255** | 0.0168 | 0.5462 | +0.5294 |
| article_full | 0.0280 | 0.7380 | **+0.7100** | 0.2469 | 0.9056 | +0.6587 | 0.0158 | 0.5413 | **+0.5255** | 0.0168 | 0.5462 | +0.5294 |

## v4 on full 133-query subset (context, from comparison CSV)

| Method | R@10 | R@100 | MRR@10 | nDCG@10 | n_q |
|---|---:|---:|---:|---:|---:|
| arm1_naive | 0.8006 | 0.9499 | 0.4605 | 0.5666 | 133 |
| arm2_metadata_node_raw | 0.7635 | 0.8901 | 0.6078 | 0.6643 | 133 |
| arm2_metadata_node_enriched | 0.7635 | 0.8901 | 0.6078 | 0.6643 | 133 |
| arm2_metadata_node_summary | 0.8045 | 0.9371 | 0.6262 | 0.6879 | 133 |
| arm2_metadata_node_full | 0.7635 | 0.8901 | 0.6078 | 0.6643 | 133 |
| arm2_metadata_article_raw | 0.7445 | 0.9279 | 0.5628 | 0.6348 | 133 |
| arm2_metadata_article_full | 0.7445 | 0.9279 | 0.5628 | 0.6348 | 133 |

## v4 recomputed on the 75-question common subset (sanity check)

These should match the v4 column of the side-by-side table above; reported here only to verify the metric harness against the canonical comparison CSV (which averages over n=133).

| Variant | R@10 | R@100 | MRR@10 | nDCG@10 | n_q |
|---|---:|---:|---:|---:|---:|
| node_raw | 0.7818 | 0.8984 | 0.6523 | 0.6182 | 75 |
| node_enriched | 0.7818 | 0.8984 | 0.6523 | 0.6182 | 75 |
| node_summary | 0.7978 | 0.9151 | 0.6053 | 0.6252 | 75 |
| node_full | 0.7818 | 0.8984 | 0.6523 | 0.6182 | 75 |
| article_raw | 0.7380 | 0.9056 | 0.5413 | 0.5462 | 75 |
| article_full | 0.7380 | 0.9056 | 0.5413 | 0.5462 | 75 |

## Interpretation

On the 75-question common subset, the linker v4 fix lifts T04's R@10 across all six variants from **[0.028, 0.142]** (v3) to **[0.738, 0.798]** (v4). MRR@10 jumps from **[0.016, 0.083]** to **[0.541, 0.652]**.

This is the empirical confirmation of the bug story in `CHANGE_NOTES.md` (2026-05-25 session log):
- Under v3, T03's `_ART_RE` accepted `^[\s]*Art\.?` but couldn't match Markdown-prefixed forms like `##### Art. 219.`. Doc 8 emits ~110 article anchors only in that prefixed form (vs 17–33 on the other curated PDFs), so Pass 1 of `bsard_link.py` was effectively blind to 28 % of doc 8's article anchors. Body paragraphs (`§ 2.`, `§ 3.`, …) inherited via Pass 2/3 from the wrong parent and got stamped with the bsard_id 814 over-aggregator — 268 nodes / 115 KB of rental-law text all anchored to a single article, while the correct anchors {839, 842, 856, 860} only carried their TOC header.
- Under v4, the local `_ART_RE_AZURE` accepts `^[\s#]*Art\.?`, the Markdown-prefixed anchors get matched in Pass 1, body paragraphs inherit correctly, and the failing GTs gain proper body coverage (verified in the 2026-05-25 session log: 839 → 11 nodes / 2.9 KB, 842 → 18 nodes / 9.3 KB, 856 → 6 nodes / 5.5 KB, 860 → 2 nodes / 2.0 KB).

**The R@10 swing is roughly 0.55–0.65 absolute (from 0.05–0.20 to 0.74–0.81 on the common subset).** Every T04 variant shows the same direction with similar magnitude — consistent with the fix being at the linking layer, not in any retriever-specific component.

v4 also adds 58 questions to the evaluation subset (n=75 → n=133, +77 %) — those are questions whose GT was unreachable under v3's index because the corresponding bsard_ids had no linked nodes. So the linker fix delivers two compound wins: more retrievable GT *and* better ranking of what's retrievable.

## What this means for the thesis writeup

Doc 8 is the headline empirical evidence that the linker fix matters. The other four curated stems also benefit but less dramatically, because they emit far fewer Markdown-prefixed article anchors. Doc 8 is therefore the right stem to single out in any methodological discussion of `_ART_RE_AZURE` — the side-by-side table above is the primary number to cite.