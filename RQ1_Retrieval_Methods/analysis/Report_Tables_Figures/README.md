# Report tables & figures

Every figure (`figures/*.pdf`, `*.png`) and LaTeX table (`tables/*.tex`) used in
the thesis report, plus the scripts that generate them. The rendered artifacts
are committed so the report builds without re-running anything; the scripts are
here so any artifact can be regenerated from the canonical result JSONs.

## Layout

```
Report_Tables_Figures/
  load_results.py     # tidy long/wide loader over output/results/ (the only place
                      # that knows the on-disk result-JSON schema)
  thesis_style.py     # single source of truth for palette, per-system styling,
                      # rcParams profiles, and save_figure / save_table helpers
  STYLEGUIDE.md       # the styling spec thesis_style.py implements
  build_rq3_all.py    # orchestrator: rebuilds every RQ3 (Ch. 14) artifact
  scripts/            # one build_*.py per table / figure group
  figures/            # generated .pdf (vector, for LaTeX) + .png (300 dpi)
  tables/             # generated .tex (tabular fragments)
```

## Data dependencies

The scripts read the gitignored result JSONs under the project data root
(`BSARD_DATA_DIR`, else `<project>/output`; see `evaluation/paths.py` and
`scripts/download_data.py`). `load_results.py`-based scripts resolve this root
automatically. The RQ3 scripts additionally read per-query sidecars and
`evaluation/data/tier3_subset.json`, and assume the default `<project>/output`
layout. This folder holds RQ1 and RQ3 artifacts only; RQ2 lives in its own
project.

## Regenerating

From the project venv (which has `matplotlib`, `pandas`, `scipy`,
`scikit-learn` per `requirements.txt`):

```bash
# a single artifact
python analysis/Report_Tables_Figures/scripts/build_t1_sparse_table.py

# every RQ3 (Chapter 14) figure + table
python analysis/Report_Tables_Figures/build_rq3_all.py
```

Each script writes into `figures/` and/or `tables/` and prints what it wrote.
Reruns are deterministic — they reproduce the committed artifacts.
