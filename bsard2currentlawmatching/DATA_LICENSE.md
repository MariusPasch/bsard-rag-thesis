# Data license & attribution

The **source code** in this repository is released under the MIT License
(see [`LICENSE`](LICENSE)).

The **data artefacts** — distributed separately via the companion Hugging Face
dataset, not in this git repository — are **derivative works** of the Belgian
Statutory Article Retrieval Dataset (BSARD) and are therefore bound by BSARD's
license:

> **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
> (CC BY-NC-SA 4.0)** — https://creativecommons.org/licenses/by-nc-sa/4.0/

This covers everything published in the corpus dataset, including:

- `bsard_corpus.db` and `bsard_corpus_clean.db` (SQLite corpora + FTS5 index)
- the Parquet snapshots (`bsard_articles_clean.parquet`, `bsard_articles_dedup.parquet`,
  `bsard_hf_articles.parquet`) and any other corpus exports
- `bsard_full_verify.csv` (source URL + verification metadata)
- the id-mapping (`*_id_mapping.json`) and corpus-statistics (`corpus_stats.json`) files
- the per-question extraction analysis under `question_analysis/`

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

The 49 source PDFs are consolidated Belgian legislation published by the
Belgian Official Journal via the Justel database (https://www.ejustice.just.fgov.be).
Belgian legal texts are official acts; this project redistributes only the
derived, BSARD-aligned corpus, not the raw PDFs, under the terms above.

If in doubt about a particular file, treat it as CC BY-NC-SA 4.0.
