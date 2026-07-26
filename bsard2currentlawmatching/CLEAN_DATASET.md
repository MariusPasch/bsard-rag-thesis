# Clean Dataset — Deduplicated PDF-Only Companion Corpus

**Author:** Marios Paschalidis
**Created:** 2026-04-23
**Companion to:** [output/bsard_corpus.db](output/bsard_corpus.db) (the main corpus)
**Build script:** [pipeline/build_clean_dataset.py](pipeline/build_clean_dataset.py)

---

## 1. Purpose

The main corpus [output/bsard_corpus.db](output/bsard_corpus.db) contains **every article record produced by the pipeline**, including:

- Multiple physical occurrences of the same article in the same PDF (e.g. table-of-contents phantom + real body + regional-variant block).
- Phase B append stubs — BSARD benchmark articles that Phase A did not find in any PDF, inserted with HuggingFace canonical text only and no PDF metadata.

This is correct behaviour for the main corpus: it preserves every signal the pipeline produced, and downstream tooling such as the citation graph and BSARD question ground-truth depend on that completeness.

**This clean dataset strips both of those to a single row per physical article**, so that analysis or retrieval work that needs *unique* articles has a non-inflated corpus to operate on. It is a read-only **companion** — the main corpus is never modified.

### When to use which dataset

| Use the **main** corpus when… | Use the **clean** corpus when… |
|---|---|
| Computing Recall@k / MRR / NDCG against the full 22,633-article BSARD benchmark | Building graph-RAG citation neighbourhoods where duplicate article nodes would distort shortest-path / centrality measures |
| You need the full citation graph edge list as-built | Reporting "number of distinct articles" / "number of distractors" — the main corpus overstates both by ~35% |
| You need every BSARD question's ground-truth article present (including Code des Sociétés) | Training/fine-tuning embeddings where duplicate documents cause memorisation |
| You want the FTS5 index over every text occurrence | Computing corpus statistics (length distributions, Jaccard overlap) without TOC phantom noise |

**Critical caveat:** 5,944 BSARD benchmark articles are absent from the clean dataset because Phase A could not extract them (see §4). If a BSARD question references one of those articles as ground truth, the retriever cannot recall it against the clean corpus. Always evaluate against the main corpus for reported benchmark numbers.

---

## 2. How it was built

### 2.1 Filter — keep only Phase A PDF extractions

Rows are retained where `pdf_page_start IS NOT NULL`. This signal is exclusive to Phase A: the main-pipeline extractor ([pipeline/extract_articles_from_pdf.py](pipeline/extract_articles_from_pdf.py)) sets a page number for every article it finds, whereas Phase B append stubs ([pipeline/link_bsard.py:401-432](pipeline/link_bsard.py#L401-L432)) explicitly set `pdf_page_start = None`.

**Rows excluded:** 5,944 Phase B stubs (14.8% of the source corpus).

Of those 5,944: 1,194 are Code des Sociétés et des Associations (no PDF exists on Justel for this code), and ~4,750 are articles from other codes where Phase A extracted the article under a different article-number spelling than the BSARD CSV (the three-fallback linker couldn't bridge Walloon / CoBAT numbering variants like `I.1-1` ↔ `I.1.1`).

### 2.2 Deduplicate by `(pdf_filename, article_number)`

Within each duplicate group, the surviving row is chosen by a tuple score (higher wins):

1. `has_text` — prefer rows with a non-empty `article_text`.
2. `is_bsard` — prefer BSARD-linked rows so the `bsard_id` join key is preserved.
3. `text_len` — prefer longer text.
4. `pdf_page_start` — prefer later pages; the body appears after the TOC in Justel PDFs, so this pushes TOC phantoms out.

**Rows collapsed:** 5,470 duplicate rows (16.0% of the 34,287 Phase A rows).

**Worked example — `Art. 1333` Code Civil Book 3:**

| Source row | Page | Source type | Text | Outcome |
|---|---|---|---|---|
| `article_id = 1616` | 2 (TOC) | `pdf_extraction_failed` | NULL | dropped |
| `article_id = 1890` | 17 (body) | `pdf_extracted` | `<Abrogé par L 2019-04-13/28 …>` | **kept** |

### 2.3 `article_id` preservation

The surviving row retains its original `article_id` from the main corpus. The ID space is therefore **non-contiguous** in the clean DB (gaps exist wherever a duplicate or stub row was dropped), but the benefit is that **every artefact keyed on `article_id` in the main corpus joins directly to the clean DB** — questions, citation edges, notebook analyses, etc.

---

## 3. Contents

### 3.1 Files

The SQLite DB and Parquet snapshot are published on Hugging Face (`mpaschalidis/bsard2currentlawmatching`) and land in `output/` via `python scripts/download_from_hf.py`. The JSONL export is not republished — regenerate it locally from the DB by running `python pipeline/build_clean_dataset.py`.

| Path | Size | Source | Description |
|---|---|---|---|
| `output/bsard_corpus_clean.db` | ~71 MB | Hugging Face | SQLite database, identical schema to [output/bsard_corpus.db](output/bsard_corpus.db) |
| `output/bsard_articles_clean.parquet` | ~10 MB | Hugging Face | Flat Parquet export |
| `output/bsard_articles_clean.jsonl` | ~63 MB | regenerate locally | JSONL export, one record per line, UTF-8 |

### 3.2 Tables

Only the `articles` table is materialised in this companion DB. The `questions` and `citation_graph` tables remain in the main corpus — with `article_id` preserved, joining back for either is a one-liner (see §5).

The `articles` schema is identical to the main corpus — see [README.md §Database Schema](README.md) or the `CREATE TABLE` statement in [pipeline/build_database.py](pipeline/build_database.py) for the 36 columns.

### 3.3 Indices

The clean DB ships with the same covering indices as the main corpus plus an FTS5 virtual table over `article_text`:

| Name | Columns |
|---|---|
| `idx_clean_bsard_id` | `bsard_id` |
| `idx_clean_law_code` | `law_code` |
| `idx_clean_pdf_file` | `pdf_filename` |
| `idx_clean_is_bsard` | `is_bsard_article` |
| `idx_clean_art_status` | `article_status` |
| `articles_fts` | Virtual FTS5 over `article_text`, `tokenize='unicode61'` |

---

## 4. Coverage statistics

### 4.1 Headline

| Metric | Clean | Main | Delta |
|---|---|---|---|
| Total rows | **28,817** | 40,231 | −11,414 (−28.4%) |
| BSARD-linked rows | 23,237 (80.6%) | 33,741 (83.9%) | −10,504 |
| Non-BSARD rows | 5,580 (19.4%) | 6,490 (16.1%) | −910 |
| Unique `bsard_id` | **16,689** | 22,633 | **−5,944** |
| Rows with `article_text IS NOT NULL` | 28,097 (97.5%) | 38,893 (96.7%) | — |
| Rows with NULL text | 720 (2.5%) | 1,338 (3.3%) | −618 |
| Unique law codes | 31 | 34 | −3 (Code des Sociétés + 2 edge codes) |

### 4.2 BSARD benchmark coverage

| | Count | % of benchmark |
|---|---|---|
| BSARD articles in clean dataset | 16,689 | **73.7%** |
| BSARD articles only in main corpus (Phase A misses) | 5,944 | 26.3% |
| BSARD benchmark total | 22,633 | 100.0% |

The 26.3% gap is the hard ceiling on retrieval Recall if the clean dataset is used as the sole corpus for benchmark evaluation — hence the caveat in §1.

### 4.3 `article_text_source` distribution

| Source | Clean | Notes |
|---|---|---|
| `bsard_dataset` | 23,237 | HuggingFace canonical text (overrides PDF text for BSARD-linked articles) |
| `pdf_extracted` | 4,860 | PyMuPDF-extracted text for non-BSARD articles |
| `pdf_extraction_failed` | 720 | Non-BSARD articles whose only PDF content was a range header or `<Abrogé par …>` marker captured as a different row; the surviving row here genuinely has no body |

### 4.4 Per-law-code breakdown (all 31 codes)

| Law code | Total | BSARD | Non-BSARD |
|---|---|---|---|
| Code Réglementaire Wallon de l'Action sociale et de la Santé | 3,684 | 3,216 | 468 |
| Code Judiciaire | 2,946 | 2,542 | 404 |
| Code Civil | 2,800 | 2,229 | 571 |
| Code de Droit Economique | 2,766 | 2,019 | 747 |
| Code de la Démocratie Locale et de la Décentralisation | 2,251 | 2,012 | 239 |
| Code du Bien-être au Travail | 1,713 | 1,593 | 120 |
| Code Wallon de l'Action sociale et de la Santé | 1,478 | 1,302 | 176 |
| Code de la Navigation | 1,447 | 940 | 507 |
| Code Pénal | 971 | 775 | 196 |
| Code d'Instruction Criminelle | 914 | 589 | 325 |
| Code Wallon du Développement Territorial | 810 | 503 | 307 |
| Code de la Fonction Publique Wallonne | 765 | 697 | 68 |
| Code Wallon de l'Enseignement Fondamental et de l'Enseignement Secondaire | 716 | 488 | 228 |
| Code Bruxellois de l'Aménagement du Territoire | 594 | 466 | 128 |
| Code Wallon de l'Agriculture | 557 | 400 | 157 |
| Code Wallon de l'Habitation Durable | 555 | 387 | 168 |
| Code Pénal Social | 456 | 389 | 67 |
| Code Bruxellois du Logement | 447 | 442 | 5 |
| Code Ferroviaire | 398 | 301 | 97 |
| Codes des Droits et Taxes Divers | 382 | 196 | 186 |
| Code Electoral | 346 | 240 | 106 |
| Code de l'Eau intégré au Code Wallon de l'Environnement | 334 | 257 | 77 |
| Code Bruxellois de l'Air, du Climat et de la Maîtrise de l'Energie | 322 | 213 | 109 |
| Code Forestier | 256 | 241 | 15 |
| La Constitution | 239 | 229 | 10 |
| Code de Droit International Privé | 156 | 147 | 9 |
| Code Rural | 153 | 107 | 46 |
| Code Wallon du Bien-être des animaux | 134 | 106 | 28 |
| Code Consulaire | 116 | 110 | 6 |
| Code Wallon de l'Environnement | 67 | 67 | 0 |
| Code de la Nationalité Belge | 44 | 34 | 10 |

Codes **absent** from the clean dataset: Code des Sociétés et des Associations (no PDF on Justel), and any other code whose every BSARD article came in as a Phase B stub.

---

## 5. Joining back to the main corpus

Because `article_id` is preserved, the clean DB joins directly to anything keyed on it in the main corpus. Open both DBs with `ATTACH`:

```sql
-- From the clean DB, attach the main corpus.
ATTACH 'output/bsard_corpus.db' AS main;
```

Then:

```sql
-- 1. BSARD questions → clean articles only.
SELECT q.question_id, q.question_text, a.article_id, a.law_code, a.article_text
FROM main.questions AS q
CROSS JOIN json_each(q.relevant_article_ids) AS rel
JOIN articles AS a ON a.article_id = CAST(rel.value AS INTEGER);

-- 2. Filter the citation graph to edges with both endpoints in the clean set.
SELECT e.*
FROM main.citation_graph AS e
WHERE e.source_id IN (SELECT article_id FROM articles)
  AND e.target_id IN (SELECT article_id FROM articles);

-- 3. Find BSARD articles that are ONLY in the main corpus (the 5,944 stubs).
SELECT a.bsard_id, a.law_code, a.article_number
FROM main.articles AS a
WHERE a.is_bsard_article = 1
  AND a.article_id NOT IN (SELECT article_id FROM articles);
```

From Python:

```python
import sqlite3

conn = sqlite3.connect("output/bsard_corpus_clean.db")
conn.execute("ATTACH 'output/bsard_corpus.db' AS main")
df = pd.read_sql("""
    SELECT a.*, q.question_text
    FROM articles a
    JOIN main.questions q
      ON a.article_id IN (SELECT value FROM json_each(q.relevant_article_ids))
""", conn)
```

From pandas alone (using the Parquet export):

```python
import pandas as pd

clean = pd.read_parquet("output/bsard_articles_clean.parquet")
full  = pd.read_parquet("output/bsard_articles.parquet")

# Semi-join: rows from the full corpus that survived the clean pass.
survivors = full[full["article_id"].isin(clean["article_id"])]
```

---

## 6. Regenerating the clean dataset

The clean dataset is purely derived from [output/bsard_corpus.db](output/bsard_corpus.db). Re-run the build script whenever the main corpus changes:

```bash
.venv/Scripts/activate          # Windows
python pipeline/build_clean_dataset.py
```

The script is idempotent — it drops `output/bsard_corpus_clean.db` if present and rebuilds from scratch. Parquet and JSONL exports are overwritten.

Options:

```bash
# Read from / write to non-default paths
python pipeline/build_clean_dataset.py \
  --src output/bsard_corpus.db \
  --dst-db output/my_clean.db \
  --dst-parquet output/my_clean.parquet \
  --dst-jsonl output/my_clean.jsonl
```

The script takes ~20 seconds on the reference 40k-row corpus and writes ~140 MB total (SQLite + Parquet + JSONL).

---

## 7. Known limitations

1. **BSARD benchmark coverage is 73.7%** — see §4.2. Do not use the clean corpus as the sole index for headline Recall@k numbers.
2. **Citation graph is not re-built** — the main corpus's `citation_graph` table references `article_id` values, some of which point to dropped duplicate rows. Edges ending on dropped IDs are still present in the main table; filter them out at query time as shown in §5.
3. **720 rows have NULL `article_text`** — these are legitimate corpus entries (range headers for abrogated article blocks, e.g. `"Art. 172-179"`) with no body text in the source PDF. Filter with `WHERE article_text IS NOT NULL` if your workload doesn't need them.
4. **`article_text_source = 'bsard_dataset'` rows contain HuggingFace canonical text, not PDF-extracted text.** The dedup rule prefers the BSARD-linked row on ties, so most BSARD articles in the clean set show `bsard_dataset` as their source even though Phase A did find them in a PDF. If you specifically need PDF-extracted raw text, pull from the `pdf_extracted`/`pdf_extraction_failed` rows in the main corpus instead.

---

## 8. Background

This dataset was produced after an internal feasibility investigation of extraction coverage which identified:

- The main corpus contains 14,308 duplicate rows keyed on `(pdf_filename, article_number)` — roughly 35% of the 40,231 total. Most are TOC-phantom pairs of BSARD articles.
- The pipeline is already at or above the practical extraction ceiling for these 49 PDFs; there is no meaningful set of articles the pipeline has missed that a different approach would recover.

The recommended "phantom-duplicate cleanup" from that investigation is what this companion dataset implements.
