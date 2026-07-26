# Data license & attribution

The **source code** in this repository is released under the MIT License
(see [`LICENSE`](LICENSE)).

The **data artefacts** — distributed separately via the companion Hugging Face
datasets, not in this git repository — are **derivative works** of the Belgian
Statutory Article Retrieval Dataset (BSARD) and are therefore bound by BSARD's
license:

> **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
> (CC BY-NC-SA 4.0)** — https://creativecommons.org/licenses/by-nc-sa/4.0/

This covers every data bundle referenced by the components of this mono-repo —
the corpus databases and exports (`bsard2currentlawmatching/`), and the cached
embeddings, BM25 stores, and result records of the retrieval projects
(`RQ1_Retrieval_Methods/`, `RQ2_Structure_Aware_Retrieval/`, `RQ3_Autonomous_Evaluation/`). Each component's
`DATA_LICENSE.md` and `DATA_CARD.md` give the per-bundle detail.

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
