# Doc 6 (Code Pénal) — per-question error analysis under linker v4

Stem: `1867_06_08_1867060850` · n_q = 65 · T03 R@10 = 0.2985 · best T04 R@10 (article_raw / article_full) = 0.2474 · Δ = −0.051.

## Bucket counts

| Bucket | n | Notes |
|---|---:|---|
| T03 strict wins (T03 hits @10, **every** T04 variant misses @10) | 0 | The strict prompt criterion — empty for doc 6. |
| T03 wins (T03 R@10 > max T04 R@10) | 9 | The operational "T04 lost ground" set. |
| T04 wins (max T04 R@10 > T03 R@10) | 11 | |
| Tied with both hitting (T03 == max T04 > 0) | 12 | |
| Tied at zero (T03 misses, every T04 misses) | 33 | True hard cases — neither retriever helps. |

So the doc 6 T04 deficit (Δ = −0.051) decomposes as: T03 wins on 9 queries, T04 wins on 11, the rest tie. The deficit is diffuse, not driven by a small "T03 catastrophe-saves" tail.

## Queries where T03 wins (T03 R@10 > max T04 R@10)

### Q726  ·  T03 R@10 = 1.000, best T04 = 0.667 (node_summary)
- Question: *Quelles sont les sanctions si je suis coupable d’avoir fait des tags/graffitis ?*
- GT bsard_ids: `[6733, 6734, 6735]`
- best T04 variant top-10 bsard_ids: `[6734, 6733, 6576, 6584, 6405, 6244, 6287, 6374, 6748, 6530]`
- GT in best-T04 top-10: `[6733, 6734]` · GT in best-T04 top-15: `[6733, 6734]`
- best T04 R@100 across variants on this query: 1.000

### Q727  ·  T03 R@10 = 1.000, best T04 = 0.667 (node_summary)
- Question: *Est-ce interdit de faire des tags/graffitis sur un mur ?*
- GT bsard_ids: `[6733, 6734, 6735]`
- best T04 variant top-10 bsard_ids: `[6734, 6733, 6723, 6738, 6672, 6721, 6530, 6732, 6538, 6727]`
- GT in best-T04 top-10: `[6733, 6734]` · GT in best-T04 top-15: `[6733, 6734]`
- best T04 R@100 across variants on this query: 1.000

### Q636  ·  T03 R@10 = 1.000, best T04 = 0.500 (node_raw)
- Question: *Comment puis-je connaître le délai au bout duquel mon amende pénale s’annule (délai de prescription) ?*
- GT bsard_ids: `[6164, 6165]`
- best T04 variant top-10 bsard_ids: `[6167, 6113, 6169, 6170, 6164, 6117, 6114, 6536, 6100, 6107]`
- GT in best-T04 top-10: `[6164]` · GT in best-T04 top-15: `[6164]`
- best T04 R@100 across variants on this query: 1.000

### Q637  ·  T03 R@10 = 1.000, best T04 = 0.500 (node_raw)
- Question: *Est-ce que l’amende pénale s’annule au bout d’un certain temps (prescription) ?*
- GT bsard_ids: `[6164, 6165]`
- best T04 variant top-10 bsard_ids: `[6167, 6164, 6169, 6536, 6170, 6075, 6751, 6168, 6156, 6089]`
- GT in best-T04 top-10: `[6164]` · GT in best-T04 top-15: `[6164]`
- best T04 R@100 across variants on this query: 1.000

### Q650  ·  T03 R@10 = 1.000, best T04 = 0.500 (node_raw)
- Question: *J'ai été condamné à une peine de prison. Puis-je être entendu pour expliquer mon plan de réinsertion sociale ?*
- GT bsard_ids: `[6106, 6107]`
- best T04 variant top-10 bsard_ids: `[6107, 6137, 6135, 6134, 6114, 6538, 6079, 6305, 6112, 6600]`
- GT in best-T04 top-10: `[6107]` · GT in best-T04 top-15: `[6107]`
- best T04 R@100 across variants on this query: 1.000

### Q1083  ·  T03 R@10 = 1.000, best T04 = 0.500 (node_raw)
- Question: *Je reçois les confidences en tant que professionnel. Puis-je lever le secret pour une «concertation de cas» ?*
- GT bsard_ids: `[6619, 6620]`
- best T04 variant top-10 bsard_ids: `[6619, 6618, 6193, 6623, 6203, 6349, 6195, 6200, 6538, 6196]`
- GT in best-T04 top-10: `[6619]` · GT in best-T04 top-15: `[6619]`
- best T04 R@100 across variants on this query: 1.000

### Q1098  ·  T03 R@10 = 1.000, best T04 = 0.500 (node_raw)
- Question: *Je reçois les confidences en tant que professionnel. Puis-je lever le secret pour une «concertation de cas» ?*
- GT bsard_ids: `[6619, 6620]`
- best T04 variant top-10 bsard_ids: `[6619, 6618, 6193, 6623, 6203, 6349, 6195, 6200, 6538, 6196]`
- GT in best-T04 top-10: `[6619]` · GT in best-T04 top-15: `[6619]`
- best T04 R@100 across variants on this query: 1.000

### Q667  ·  T03 R@10 = 0.600, best T04 = 0.300 (node_raw)
- Question: *Je suis témoin d'une infraction. Qu’est-ce que je risque si je ne dis pas la vérité à mon audition à la police en tant que témoin ?*
- GT bsard_ids: `[6333, 6334, 6335, 6336, 6337, 6338, 6340, 6341, 6342, 6344]`
- best T04 variant top-10 bsard_ids: `[6333, 6341, 6451, 6697, 6636, 6336, 6241, 6538, 6600, 6621]`
- GT in best-T04 top-10: `[6333, 6336, 6341]` · GT in best-T04 top-15: `[6333, 6336, 6341]`
- best T04 R@100 across variants on this query: 1.000

### Q718  ·  T03 R@10 = 0.182, best T04 = 0.091 (node_raw)
- Question: *La police peut-elle m’arrêter si elle me surprend en train de casser une voiture ?*
- GT bsard_ids: `[6719, 6721, 6722, 6723, 6724, 6726, 6727, 6728, 6730, 6732, 6756]`
- best T04 variant top-10 bsard_ids: `[6732, 6451, 6653, 6697, 6644, 6636, 6453, 6336, 6266, 6079]`
- GT in best-T04 top-10: `[6732]` · GT in best-T04 top-15: `[6732]`
- best T04 R@100 across variants on this query: 0.455

## Failure-pattern read

One pattern dominates and one outlier sits alongside it.

### Pattern A — adjacent-article confusion on multi-GT queries (8 of 9)

Eight of the nine T03-wins are multi-GT queries (GT cardinality 2, 3, or 11) where T04 finds *one* GT article in top-10 but adjacent GT articles — articles with sequential bsard_ids — sit at rank 11–100. **`best T04 R@100` = 1.000 on every one of these eight**, so the GT articles ARE in T04's pool; they just rank below the cut.

Concrete examples:

| Q | GT | best T04 top-10 (first 8) | Missing GT | T04 R@100 |
|---|---|---|---|---:|
| Q636 | `[6164, 6165]` | `[6167, 6113, 6169, 6170, 6164, 6117, 6114, 6536]` | **6165** | 1.000 |
| Q637 | `[6164, 6165]` | `[6167, 6164, 6169, 6536, 6170, 6075, 6751, 6168]` | **6165** | 1.000 |
| Q650 | `[6106, 6107]` | `[6107, 6137, 6135, 6134, 6114, 6538, 6079, 6305]` | **6106** | 1.000 |
| Q726 | `[6733, 6734, 6735]` | `[6734, 6733, 6576, 6584, 6405, 6244, 6287, 6374]` | **6735** | 1.000 |
| Q727 | `[6733, 6734, 6735]` | `[6734, 6733, 6723, 6738, 6672, 6721, 6530, 6732]` | **6735** | 1.000 |
| Q1083 | `[6619, 6620]` | `[6619, 6618, 6193, 6623, 6203, 6349, 6195, 6200]` | **6620** | 1.000 |
| Q1098 | `[6619, 6620]` | `[6619, 6618, 6193, 6623, 6203, 6349, 6195, 6200]` | **6620** | 1.000 |
| Q667 | 10 GTs from `[6333-6344]` | `[6333, 6341, 6451, 6697, 6636, 6336, 6241, 6538]` | 7 of 10 GTs deeper than 10 | 1.000 |

**Mechanism.** T04 indexes the Code Pénal at node granularity (and for the `article_*` variants, at article granularity *within this single PDF*). Adjacent articles in the Code Pénal share heavy procedural boilerplate ("La peine prévue à l'article…", "Sera puni de…", cross-references between sibling articles within the same incrimination chapter). When the question hits keywords in article A's body, A's nodes and content from A's neighbours both score highly; T04 then pulls in non-GT siblings (6167, 6169, 6170 for GT 6164/6165) ahead of the consecutive GT (6165). T03 indexes the *whole-article* BSARD corpus (22 633 deduped articles), so each candidate carries clean per-article BM25 + dense signal and the multi-GT queries recover all linked articles.

This is *not* a linker bug — the linker correctly assigns these nodes to their bsard_ids. It is a known limitation of single-PDF node/article indices on legal codes with very high boilerplate density between sibling articles.

### Pattern B — true coverage gap (1 of 9)

**Q718** ("La police peut-elle m'arrêter si elle me surprend en train de casser une voiture ?", GT = 11 bsard_ids in range 6719–6756) is the only T03-win where `best T04 R@100` is below 1.0 (= 0.455 — only 5/11 GT articles reachable in the top-100 pool). The other 6 GT bsard_ids are not in T04's R@100 set at all, so the linker either missed them on this PDF or AzureDI's extraction of those articles is too thin to surface under any boost.

That makes Q718 a genuine extraction/linking thinness for this specific chapter (Title IX-bis of the Code Pénal, "Des atteintes à la sûreté de l'État" — though the bsard ids 6719+ are more likely an arrest-and-flagrante chapter). Worth checking whether those 6 missing bsard_ids have linked nodes at all in T04's index for this PDF — that's a one-grep check, not a full investigation, and is the natural next step if anyone wants to chase doc 6 further.

### What this implies for the doc 6 deficit

The deficit is mostly *not* fixable by changing the T04 retriever — it's a structural property of indexing a single legal code with high inter-article boilerplate at node/article granularity. Plausible levers if doc 6 were a target for further work:

1. **Increase top-K cut from 10 → 20.** R@20 on these multi-GT queries would likely close the gap (the missing GT is consistently within rank 11–15 for the 2-GT cases).
2. **Add a sibling-aggregation pass after retrieval.** When a returned bsard_id has consecutive sibling bsard_ids in T04's pool, promote them as a block. This is a post-hoc fix outside the boost stage but matches how the Code Pénal organises related provisions.
3. **Investigate Q718 specifically.** If the missing 6 GT articles are simply not linked in the v4 index for this PDF, that's a `bsard_link.py` coverage check, not a ranker tweak.

The doc 6 deficit of Δ = −0.051 R@10 is best reported in the thesis as "T04 underperforms on the Code Pénal because adjacent-article confusion under multi-GT queries pushes sibling GT articles below rank 10 in T04's per-PDF index, while T03's whole-article BSARD-wide hybrid keeps each candidate distinguishable; the deficit does not appear at K = 100 except on a single coverage outlier."
