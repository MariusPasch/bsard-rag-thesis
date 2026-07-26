# RAG Visualization — Demo & Review Guide

How to launch the viewer for the thesis presentation, which PDF + question to use
for each arm, and what to look for. All commands are PowerShell, run **from the
RQ2_Structure_Aware_Retrieval repo root**.

The viewer is read-only: it overlays each arm's cached retrieval results on the
original PDF page layout. Switching the **Arm** dropdown re-runs the *same*
question through a different arm, so you can load one bundle and flip between
arms to compare.

---

## 0. Two rules

Run every command **from the RQ2_Structure_Aware_Retrieval repo root**. Two things
*must* be right or the demo looks broken:

| Requirement | Why |
|---|---|
| Use **`.\.venv\Scripts\streamlit.exe`** (the project venv), not bare `streamlit` | The system Python lacks `faiss`/`torch`; the venv has everything. |
| Always pass **`--db "...\bsard_corpus.db"`** | Article numbers + canonical text are resolved from the DB. Without it every article shows *"no DB entry"* (not an arm bug — just a missing DB). |

The dev-mode file watcher is already disabled (`.streamlit/config.toml`), so the
terminal stays quiet. Editing source no longer auto-reloads — restart to pick up
changes.

---

## 1. Verified rank matrix

First-hit rank of a gold article per arm (lower = better; **✗** = miss@10).
Measured from each arm's cached results. Use the **green rows** when you want
every arm to show correctly-retrieved articles.

| Bundle | PDF | ARM1 | 2A | 2B | 2C | GT size | Good for |
|---|---|:--:|:--:|:--:|:--:|:--:|---|
| **q602** | Code Judiciaire (larger) | 1 | 1 | 1 | 1 | 11 | ✅ all arms — flagship |
| **q973** | Code du Logement | 1 | 3 | 4 | 1 | 6 | ✅ all arms |
| **q649** | Code Pénal | 1 | 1 | 1 | 2 | 2 | ✅ all arms — simple |
| q192 | Code Civil | 1 | 1 | **11✗** | 2 | 3 | 2B failure case / canonical 2C |
| q865 | Code du Logement | 4 | **9** | 1 | 2 | 4 | 2A-weak case |
| q101 | Code Judiciaire (larger) | 10 | 2 | **46✗** | 1 | 6 | 2C-shines case |
| q711 | Code Judiciaire (larger) | **11✗** | 2 | 8 | **27✗** | 3 | 2C failure case |

---

## 2. Launch — all-arm demos (every arm retrieves correctly)

Use these when you want to flip through all four arms and have each one show
hits. Each command is **self-contained** — paste the whole block (one line; the
`#` lines are comments). After it opens, set **Mode → retrieval** and pick the
**Arm** from the top-bar dropdown.

```powershell
# FLAGSHIP — all four arms rank 1. Code Judiciaire, "Qu'est-ce qu'une citation ?"
# GT = Arts 4787–4797 (an 11-article contiguous block) — paints as one big green band.
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\1967_10_10_1967101055\t08_bundles\483beb56a47b\q602.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

```powershell
# ALL ARMS HIT — Code du Logement, "bail officiel de colocation à Bruxelles ?"
# GT = 611 / 878–882.  ARM1 1 · 2A 3 · 2B 4 · 2C 1.
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\2003_07_17_2013A31614\t08_bundles\483beb56a47b\q973.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

```powershell
# ALL ARMS HIT (simple, GT=2) — Code Pénal, electronic surveillance.
# GT = 6106 / 6107.  ARM1 1 · 2A 1 · 2B 1 · 2C 2.
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\1867_06_08_1867060850\t08_bundles\50744bfce1d0\q649.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

## 3. Launch — contrast cases (one arm intentionally fails)

The other arms still retrieve correctly; the named arm is the teaching point.

```powershell
# 2B FAILS (rank 11) while ARM1·2A·2C hit — Code Civil, marriage annulment (foreigner).
# Also the canonical ARM2C sample. GT = 920 / 1014 / 1045.  ARM1 1 · 2A 1 · 2C 2 · 2B 11✗.
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\1804_03_21_1804032150\t08_bundles\483beb56a47b\q192.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

```powershell
# 2C SHINES / 2B FAILS — Code Judiciaire, "recours contre une décision d'admissibilité ?"
# GT = 5152–5154 / 5229–5231.  2C 1 (8-step descent) · 2A 2 · ARM1 10 · 2B 46✗ (deep nav beats flat ToC nav).
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\1967_10_10_1967101055\t08_bundles\50744bfce1d0\q101.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

```powershell
# 2A WEAK (rank 9) — Code du Logement, "bail 9 ans à Bruxelles ?"  (nav arms win)
# GT = 611 / 614 / 856 / 857.  2B 1 · 2C 2 · ARM1 4 · 2A 9.
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\2003_07_17_2013A31614\t08_bundles\483beb56a47b\q865.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

```powershell
# 2C FAILS (review) — Code Judiciaire, "Qui doit payer les honoraires de l'avocat ?"
# GT = 5119–5121.  2A 2 · 2B 8 hit; 2C MISS (rank 27 after a 28-step wander), ARM1 also misses (11).
.\.venv\Scripts\streamlit.exe run "RQ2_T08_RAG_VISUALIZATION\src\visualization\app.py" -- --bundle-json "RQ2_T03_ARM1_NAIVE\data\1967_10_10_1967101055\t08_bundles\50744bfce1d0\q711.json" --db "$RQ2_DATA_DIR/bsard_corpus.db"
```

> Tip: to shorten, set `$DB = "$RQ2_DATA_DIR/bsard_corpus.db"` once, then use `--db $DB`. Only works in the *same* shell where you set it.

---

## 4. Best pick per arm

For each arm: a **presentation** pick (it shines) and a **review** pick
(instructive weakness). Load the bundle, set the Arm dropdown, read the
right-hand panels.

### ARM1 — Naive Chunking
- **Present:** `q602` (Code Judiciaire), ARM1 rank 1. The GT is one long run of
  articles (4787–4797); the ranked sliding-window chunks tile across the whole
  section — good for showing chunk granularity vs article boundaries. Retrieval
  is **live** (FAISS + BM25), so the first run loads the embedding model (a few
  seconds).
- **Review:** any bundle in **Mode → review** → click **Re-chunk**. The chunk
  grid paints over the whole PDF — shows where a single article is split across
  windows, or where one window straddles two articles.
- **What to expect:** green = chunk overlapping a GT article, red = chunk that
  doesn't. Sidebar → *"Articles in this chunk"* lists the BSARD articles each
  chunk covers.

### ARM2A — Metadata-Enrichment-Summary
- **Present:** `q602` (Code Judiciaire) or `q649` (Code Pénal) — both rank 1.
  (`q192` Code Civil is also rank 1 and beats the nav arms.)
- **Review:** `q865` (Code du Logement). Here 2A is **weak (rank 9)** while the
  navigation arms win — good for the "metadata boost is inert on this doc" point.
- **What to expect:** article/node-level hits; a node hit also draws a thin
  **page-band** marker on its specific page. Right panel → *"Retrieval signals
  (2A)"* shows variant / unit / boosts.

### ARM2B — PageIndex
- **Present:** `q602` (Code Judiciaire), rank 1. The LLM ToC navigation lands the
  right chapter and returns the whole "citation" section.
- **Review:** `q192` (Code Civil), 2B **misses @10 (rank 11)** — a wrong-chapter
  navigation; or `q101` (rank 46) for a starker miss. Open the sidebar **ToC
  tree** explorer + right panel *"Navigation trace (2B)"* to see which chapter it
  wrongly entered and why.
- **What to expect:** sidebar **ToC tree** (selected law 🟦 / chapter 🟧 /
  article 🟢); per-step LLM selections in the trace panel.

### ARM2C — AzureDI-Agentic-Tree-Navigation-CNR
- **Present:** `q101` (Code Judiciaire). A ReAct agent descends the **depth-7**
  deep tree in 8 steps and ranks the gold block at **rank 1**, while flat ToC nav
  (2B) misses badly (rank 46) — the cleanest "deep agentic nav wins" case.
  (`q192` Code Civil is the canonical verified sample: 9 steps, gold at rank 2.)
- **Review:** `q711` (Code Judiciaire). 2C **misses** — it wanders for **28
  steps** and still ranks the gold (5119–5121) at 27, while 2A (rank 2) and 2B
  (rank 8) find it. Open the trace + deep-tree explorer to see the agent
  over-exploring the wrong branches.
- **What to expect:** sidebar **Deep tree** explorer auto-expands the descent
  path (🔍) and bolds selected leaves (🟢); right panel → *"ReAct navigation
  trace (2C)"* shows, per step, the node title → the LLM's *thought* → which
  sub-sections it descended into and which articles it selected.

---

## 5. Three cross-arm story slides

Load the bundle once, then flip the Arm dropdown left-to-right:

| Bundle | Story | Ranks (ARM1 · 2A · 2B · 2C) |
|---|---|---|
| `q602` Code Judiciaire | **All four arms succeed** — the triumphant baseline | 1 · 1 · 1 · 1 |
| `q192` Code Civil | **Navigation can go wrong** — chunking, metadata & deep-nav win; flat ToC nav fails | 1 · 1 · **11✗** · 2 |
| `q865` Code du Logement | **Structure beats keywords** — nav arms win, metadata weak | 4 · **9** · 1 · 2 |

---

## 6. Review-mode notes

- Toggle **Mode → review** (top bar) to see *what each arm indexed*, independent
  of any question: ARM1 shows chunks (click **Re-chunk**); the Arm-2 arms show
  article/node spans.
- Selecting an item in the sidebar paints it on the PDF and shows its text in the
  right-hand **Selected** panel. (Keep `--db` on so article text/numbers resolve.)
- Review mode is read-only w.r.t. the pipeline — nothing you click writes back to
  indices, results, or ground truth.

---

## 7. Notes / gotchas

- Ranks above are the **first** gold article hit; with multi-article GT (e.g.
  q602's 11) several greens paint at once.
- **Code Civil has only one bundle** (`q192`). The other four PDFs have many —
  browse them under `RQ2_T03_ARM1_NAIVE\data\<stem>\t08_bundles\<qset>\`.
- A bundle only carries one PDF's `article_spans`, so a given question can only
  be viewed on its own PDF.
- If an arm isn't in the dropdown for a bundle, that arm has no cached result for
  that question — pick another question or arm.
- Regenerate the ARM2C data after re-running its experiment:
  `python RQ2_T04_ARM2_METADATA\experiments\arm2c\export_for_t08.py`.
