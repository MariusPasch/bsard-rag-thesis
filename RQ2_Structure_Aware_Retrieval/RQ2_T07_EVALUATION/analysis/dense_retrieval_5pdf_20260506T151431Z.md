# Dense-only retrieval — 5 curated PDFs

Generated 2026-05-06 15:14Z. Reranking: candidates sorted by `dense_score` (FAISS only; sparse/fused ignored). Chunk → bsard_id aggregation by max dense score per question. Layer 2 drift filter is applied: every chunk's `bsard_ids` is restricted to the IN_PDF set, so the dense run can only ever surface bsard_ids whose canonical text is genuinely in the PDF.

Evaluation is over the **full BSARD question set** (1108 questions = the union of BSARD's train and test splits). The split is purely a BSARD benchmark convention; RQ2 does no training, so the splits are merged here to avoid implying otherwise.

**Per-question identity (for evaluable questions):** `T1/R = E/R × recall_ceiling`. Layer 2 mechanically caps T1 at the ceiling — drift is the wedge between the two metrics.

## Scope

- Corpus: **618 pages**, **3060 chunks** (sliding window 512 tokens, stride 256, e5-large-instruct).
- Articles: **2759 BSARD-linked**, of which **2613 are IN_PDF** post-drift (canonical text genuinely present in the canonical PDF).

## Headline

- Mean across the 5 PDFs: **T1/R@10 = 0.439**, **E/R@10 = 0.540**.
- Drift cost: of 813 questions linked to one of the 5 PDFs, **21** are not-evaluable on their PDF (all GT drifted out).
- Mean recall ceiling among linked questions across the 5: 0.796 — this is the per-PDF upper bound for T1 and explains the wedge to E.

- Best PDF on E/R@10: `2003_07_17_2013A31614` (0.762, n_evaluable=134).
- Worst PDF on E/R@10: `1967_10_10_1967101055` (0.408, n_evaluable=218).
- Biggest E↔T1 gap: `1867_06_08_1867060850` (E/R@10=0.466, T1/R@10=0.330, Δ=0.136; linked-ceiling=0.686).

## Per-PDF breakdown

| doc_id | pages | chunks | articles | in_pdf | n_link | n_evaluable | drifted | T1/R@10 | E/R@10 | ceiling_linked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1804_03_21_1804032150` | 129 | 626 | 540 | 515 | 265 | 264 | 1 | 0.470 | 0.568 | 0.834 |
| `1867_06_08_1867060850` | 138 | 711 | 689 | 651 | 106 | 89 | 17 | 0.330 | 0.466 | 0.686 |
| `1967_10_10_1967101055` | 173 | 854 | 895 | 862 | 219 | 218 | 1 | 0.308 | 0.408 | 0.742 |
| `1967_10_10_1967101056` | 79 | 419 | 356 | 348 | 87 | 87 | 0 | 0.419 | 0.499 | 0.863 |
| `2003_07_17_2013A31614` | 99 | 450 | 279 | 237 | 136 | 134 | 2 | 0.670 | 0.762 | 0.857 |

## Soundness note

Under Layer 2, every chunk's `bsard_ids` is a subset of IN_PDF, so any retrieved bsard_id is also in IN_PDF. For an evaluable question, `hit ∩ full_GT = hit ∩ gt_in_pdf`, hence the per-question identity `T1/R = E/R × recall_ceiling`. There can therefore be no question where E/R = 0 but T1/R > 0 — the metric is internally consistent by construction (no need for a runtime check).
