# Data license & attribution

The **source code** in this repository is released under the MIT License
(see [`LICENSE`](LICENSE)).

The **data artefacts** — distributed separately via the companion Hugging Face
dataset, not in this git repository — are **derivative works** of the Belgian
Statutory Article Retrieval Dataset (BSARD) and of the underlying Belgian
statutory legislation, and are distributed under BSARD's license:

> **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
> (CC BY-NC-SA 4.0)** — https://creativecommons.org/licenses/by-nc-sa/4.0/

This covers, to the extent each reproduces BSARD or statutory corpus text:

- `bsard_corpus.db` (SQLite corpus + index) and any corpus exports
- the curated source PDFs of Belgian statutory codes used by the retrieval arms
- the Azure Document Intelligence (`AzureDI`) layout export derived from those PDFs
  (released in minimised form — see `DATA_CARD.md` and `scripts/sanitise_azuredi.py`)
- the per-arm retrieval indices (Arm 1 FAISS + BM25 stores; Arm 2A metadata
  FAISS; Arm 2B PageIndex trees) and their cached embeddings
- the per-query ground-truth files and the experiment result JSONs under each
  arm's `results/`, to the extent they reproduce corpus text

### What CC BY-NC-SA 4.0 means here

- **Attribution** — credit BSARD (citation below).
- **NonCommercial** — these artefacts may not be used for commercial purposes.
- **ShareAlike** — redistributions/derivatives must use the same license.

### A note on the source PDFs

The source documents are official Belgian statutory codes (e.g. Code Civil,
Code Pénal, Code Judiciaire, Code du Logement). Belgian legislative texts are
public-domain official acts; the **selection, extraction, layout parsing and
chunking** applied here are project-specific derivatives of BSARD and are
covered by CC BY-NC-SA 4.0 as above. The raw legislation remains freely
available from the official Belgian sources (Justel / Moniteur belge).

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
