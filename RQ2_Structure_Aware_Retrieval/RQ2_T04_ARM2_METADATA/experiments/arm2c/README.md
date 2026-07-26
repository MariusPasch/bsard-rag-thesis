# Arm-2C — Agentic tree navigation

A ReAct agent (LLaMA 3.1 8B, vectorless) descends a **deep** AzureDI tree to select
articles. Hypothesis: Arm-2B/PageIndex underperforms because its tree is *flattened*
to ~one chapter level (~108 sibling chapters → one-shot 108-way pick → wrong-chapter
miss). Rebuilding the **full** hierarchy (LIVRE › TITRE › CHAPITRE › Section › Article)
turns that into a sequence of ~5-way choices and should recover the misses.

## Per-PDF preprocessing (the locked logic)

For each PDF: **read each article's header stack → file it under that stack → empty
stacks go to one "Unfiled" branch.** The merge of addresses *is* the tree. No
node-by-node `parent_id` repair (AzureDI's `parent_id` is a forest of ~200 roots on
the big codes; the header-stack re-nesting is what turns it into one navigable tree).

## Files

| File | Role | Where it runs |
|---|---|---|
| `deep_tree_builder.py` | raw AzureDI → `data/<doc>/deep_tree.json` (+ native-EN `level_summary`/`content_summary`/`keywords`) | **local** (T05 venv: needs arm2_metadata/arm2_pageindex/bsard DB) |
| `tree_stats.py` | honest nav-cost stats; `--write` refreshes manifests | local |
| `prepare_bundle.py` | `bundles/<doc>/{queries,baselines}.json` (qid→text+clipped gold from bsard DB; canonical Arm-1/2A/2B recall@10) | local |
| `navigator_tools.py` | `DeepTree` + tools (`expand`/`open`/`select`) + preflight + renderer; vendored `Node` | **runtime (stdlib only)** |
| `react_navigator.py` | frontier-descent ReAct loop + self-contained `LlamaClient` | runtime (stdlib + `ollama`) |
| `check_results.py` | quick recall@10 vs baselines → SUCCESS/FAILURE | runtime/local |
| `analyze_results.py` | **decision-support report**: miss decomposition + recommendation (scale vs change-what) → `analysis_report.md` + `per_query.csv` | runtime/local |
| `_build_azure_notebook.py` → `azure_arm2c_run.ipynb` | the Azure run notebook | local build / Azure run |

The **runtime trio** (`navigator_tools`, `react_navigator`, `check_results`) imports
nothing but stdlib + `ollama`, so the VM needs only this repo + `pip install ollama`.

## Selected PDF: `1804_03_21_1804032150` (Code Civil)

Best chance: most gold (252 questions), richest deep tree (depth 7, median fan-out
2.5), biggest flat→deep contrast (108→~5). Baselines on the same 252 q (recall@10):
**Arm-1 0.471 · Arm-2A 0.514 · Arm-2B 0.203** ← the bar.

## Run it (Azure compute)

1. **Local prep is already done** for Code Civil (`data/…/deep_tree.json`,
   `bundles/…/queries.json`). To target another PDF:
   `python deep_tree_builder.py --doc-id <stem>` then `python prepare_bundle.py --doc-id <stem>`.
2. **Push `experiments/arm2c/` to GitHub** (the notebook clones this repo to get the
   code + tree + bundle).
3. On the Azure compute instance, open **`azure_arm2c_run.ipynb`**, set `GITHUB_TOKEN`
   (+ `DOC_ID`/`MODE` if changing), and run cells top to bottom. It installs Ollama,
   runs the warmup → single-query → 5-query pilot + **ETA gate**, then the full run,
   then prints the **success/failure check**.
4. Re-run with `MODE = "enriched"` to test whether the native-EN summaries help
   (the mixed-language check), and point the loop at Arm-2B's flat tree for the
   flat-vs-deep ablation.

## Analyse results (decide scale vs change approach)

Notebook cell 8 runs this automatically; standalone:
```
python analyze_results.py --results <dir of q*.json> \
       --bundle bundles/<doc>/ --tree data/<doc>/deep_tree.json
```
Reports recall@k vs baselines, navigation behaviour, and the **miss decomposition** —
every gold article as HIT / **SEEN_NOT_SELECTED** (seen but not picked → selection
prompt) / **NOT_REACHED** (branch pruned → tree/navigation, the Arm-2B failure mode) /
ORPHAN_UNREACHED. The recommendation reads that decomposition: navigation-dominated
misses → fix the approach (enriched mode / MAX_NODES / summaries) before scaling;
selection-dominated → tune the prompt; healthy success → scale to the other PDFs.
Writes `analysis_report.md` + `per_query.csv`. (`check_results.py` is the quick
pass/fail.) Final cross-arm numbers should still flow through the T06 one-path eval.
