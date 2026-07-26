# Dense-only retrieval — 5 curated PDFs

Generated 2026-05-06 14:54Z. Reranking: candidates sorted by `dense_score` (FAISS only; sparse/fused ignored). Chunk → bsard_id aggregation by max dense score per question. Layer 2 drift filter is applied: every chunk's `bsard_ids` is restricted to the IN_PDF set, so the dense run can only ever surface bsard_ids whose canonical text is genuinely in the PDF.

**Per-question identity (for evaluable questions):** `T1/R = E/R × recall_ceiling`. Layer 2 mechanically caps T1 at the ceiling — drift is the wedge between the two metrics.

## TRAIN qset

- Headline (mean across the 5 PDFs): **T1/R@10 = 0.438**, **E/R@10 = 0.538**.
- Drift cost: of 650 linked questions, **18** are not-evaluable on their PDF (all GT drifted out).
- Mean recall ceiling among linked questions across the 5: 0.790 — this is the per-PDF upper bound for T1 and explains the gap to E.

- Best PDF on E/R@10: `2003_07_17_2013A31614` (0.765, n_evaluable=100).
- Worst PDF on E/R@10: `1967_10_10_1967101055` (0.407, n_evaluable=174).
- Biggest E↔T1 gap: `1867_06_08_1867060850` (E/R@10=0.430, T1/R@10=0.308, Δ=0.121; linked-ceiling=0.667).

Per-PDF breakdown (linked → evaluable, drift cost):

| doc_id | n_link | n_evaluable | drifted | T1/R@10 | E/R@10 | ceiling_linked |
|---|---:|---:|---:|---:|---:|---:|
| `1804_03_21_1804032150` | 222 | 221 | 1 | 0.473 | 0.570 | 0.837 |
| `1867_06_08_1867060850` | 78 | 64 | 14 | 0.308 | 0.430 | 0.667 |
| `1967_10_10_1967101055` | 175 | 174 | 1 | 0.300 | 0.407 | 0.733 |
| `1967_10_10_1967101056` | 73 | 73 | 0 | 0.447 | 0.520 | 0.865 |
| `2003_07_17_2013A31614` | 102 | 100 | 2 | 0.663 | 0.765 | 0.847 |

## TEST qset

- Headline (mean across the 5 PDFs): **T1/R@10 = 0.430**, **E/R@10 = 0.535**.
- Drift cost: of 163 linked questions, **3** are not-evaluable on their PDF (all GT drifted out).
- Mean recall ceiling among linked questions across the 5: 0.814 — this is the per-PDF upper bound for T1 and explains the gap to E.

- Best PDF on E/R@10: `2003_07_17_2013A31614` (0.753, n_evaluable=34).
- Worst PDF on E/R@10: `1967_10_10_1967101056` (0.393, n_evaluable=14).
- Biggest E↔T1 gap: `1867_06_08_1867060850` (E/R@10=0.557, T1/R@10=0.390, Δ=0.168; linked-ceiling=0.737).

Per-PDF breakdown (linked → evaluable, drift cost):

| doc_id | n_link | n_evaluable | drifted | T1/R@10 | E/R@10 | ceiling_linked |
|---|---:|---:|---:|---:|---:|---:|
| `1804_03_21_1804032150` | 43 | 43 | 0 | 0.451 | 0.561 | 0.816 |
| `1867_06_08_1867060850` | 28 | 25 | 3 | 0.390 | 0.557 | 0.737 |
| `1967_10_10_1967101055` | 44 | 44 | 0 | 0.341 | 0.410 | 0.777 |
| `1967_10_10_1967101056` | 14 | 14 | 0 | 0.274 | 0.393 | 0.852 |
| `2003_07_17_2013A31614` | 34 | 34 | 0 | 0.692 | 0.753 | 0.889 |

## Soundness note

Under Layer 2, every chunk's `bsard_ids` is a subset of IN_PDF, so any retrieved bsard_id is also in IN_PDF. For an evaluable question, `hit ∩ full_GT = hit ∩ gt_in_pdf`, hence the per-question identity `T1/R = E/R × recall_ceiling`. There can therefore be no question where E/R = 0 but T1/R > 0 — the metric is internally consistent by construction (no need for a runtime check).
