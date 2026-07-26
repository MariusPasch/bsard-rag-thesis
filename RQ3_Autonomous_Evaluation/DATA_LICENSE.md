# Data license & attribution

The **source code** in this repository is released under the MIT License
(see [`LICENSE`](LICENSE)).

This repository is primarily an **evaluation harness** — it computes metrics over
retrieval results produced elsewhere (see the RQ1 project). It does **not** ship
the BSARD corpus text. The only data tracked here are small aggregate result
tables and stratum summaries under `analysis/` (e.g. `system_summary.tsv`,
`strata_summary.json`).

Those result tables, and any other artefacts **derived** from the Belgian
Statutory Article Retrieval Dataset (BSARD), are **derivative works** of BSARD
and are therefore bound by BSARD's license:

> **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
> (CC BY-NC-SA 4.0)** — https://creativecommons.org/licenses/by-nc-sa/4.0/

This covers:

- the aggregate metric tables under `analysis/` (scores computed over BSARD)
- the stratum / query-subset summaries under `analysis/`
- any corpus exports, embeddings, qrels, or result JSONs that reproduce or are
  derived from BSARD content (none are committed here; see below)

### Where the corpus artefacts live

The BSARD corpus database, parquet exports, and cached embeddings are **not**
distributed in this repository. They are produced and hosted by the companion
RQ1 project — see the Hugging Face dataset
[`mpaschalidis/bsard-rq1-data`](https://huggingface.co/datasets/mpaschalidis/bsard-rq1-data),
which is likewise licensed CC BY-NC-SA 4.0.

### What CC BY-NC-SA 4.0 means here

- **Attribution** — credit BSARD (citation below).
- **NonCommercial** — these artefacts may not be used for commercial purposes.
- **ShareAlike** — redistributions/derivatives must use the same license.

### Original dataset

- Corpus & benchmark: BSARD, Maastricht Law & Tech Lab.
- Hub: https://huggingface.co/datasets/maastrichtlawtech/bsard
- Code: https://github.com/maastrichtlawtech/bsard

```bibtex
@inproceedings{louis2022statutory,
  title     = {A Statutory Article Retrieval Dataset in French},
  author    = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  year      = {2022},
  pages     = {6789--6803},
  publisher = {Association for Computational Linguistics},
}
```

If in doubt about a particular file, treat it as CC BY-NC-SA 4.0.
