"""Per-PDF status surface for the RQ2 pipeline.

Read-only. Discovers per-project artefacts, validates them against the
schema rules in PLAN_per_pdf_pipeline §C, and writes a registry file at
``RQ2_T00_ORCHESTRATOR/data/status/<doc_id>.json``. Never invokes a
project, never mutates a sibling project's data.

CLI:

    python -m orchestrator.status --doc-id <doc_id>
    python -m orchestrator.status --all
    python -m orchestrator.status --doc-id <doc_id> --json
    python -m orchestrator.status --doc-id <doc_id> --strict
    python -m orchestrator.status --doc-id <doc_id> --project T04
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import paths

SCHEMA_VERSION = 1
PROJECTS_IN_OVERALL = ("T02", "T03", "T04", "T05", "T07")
PROJECTS_INFORMATIONAL = ("T08",)
ALL_PROJECTS = PROJECTS_IN_OVERALL + PROJECTS_INFORMATIONAL

logger = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    path: str
    validator: str
    ok: bool
    detail: str | None = None
    warning: bool = False  # True = warn-only; never downgrades project status

    def to_dict(self) -> dict:
        d = {"name": self.name, "path": self.path, "validator": self.validator, "ok": self.ok}
        if self.detail is not None:
            d["detail"] = self.detail
        if self.warning:
            d["warning"] = True
        return d


# ── Output guard ─────────────────────────────────────────────────────────────


def _status_dir() -> Path:
    return paths.project_dir("T00") / "data" / "status"


def _assert_inside_status_dir(p: Path) -> None:
    """Hard read-only guard (PLAN §C.5)."""
    target = p.resolve()
    allowed = _status_dir().resolve()
    try:
        target.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(
            f"status command refuses to write outside {allowed}: {target}"
        ) from exc


# ── Validators ───────────────────────────────────────────────────────────────


def _rel(p: Path) -> str:
    """Render path relative to RQ2_ROOT, using forward slashes for portability."""
    try:
        return p.resolve().relative_to(paths.RQ2_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _read_json(p: Path) -> dict | None:
    """Read JSON; return None on any failure (caller turns this into ok=False)."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _check_exists(name: str, p: Path, *, warning: bool = False) -> CheckResult:
    if not p.exists():
        return CheckResult(name, _rel(p), "exists", False, "path not found", warning)
    return CheckResult(name, _rel(p), "exists", True, warning=warning)


def _check_non_empty(name: str, p: Path) -> CheckResult:
    if not p.exists():
        return CheckResult(name, _rel(p), "non_empty", False, "path not found")
    if p.stat().st_size == 0:
        return CheckResult(name, _rel(p), "non_empty", False, "file is zero bytes")
    return CheckResult(name, _rel(p), "non_empty", True)


def _check_json_schema(
    name: str, p: Path, *, schema_version_min: int = 1, expected_doc_id: str | None = None,
    json_path: list[str] | None = None,
) -> CheckResult:
    payload = _read_json(p)
    if payload is None:
        return CheckResult(name, _rel(p), "json_schema", False,
                           "json parse error or path missing")
    sub = payload
    if json_path:
        for key in json_path:
            if not isinstance(sub, dict) or key not in sub:
                return CheckResult(name, _rel(p), "json_schema", False,
                                   f"json path {'/'.join(json_path)} missing")
            sub = sub[key]
    sv = sub.get("schema_version") if isinstance(sub, dict) else None
    if sv is None:
        return CheckResult(name, _rel(p), "json_schema", False,
                           "schema_version field missing")
    if sv < schema_version_min:
        return CheckResult(name, _rel(p), "json_schema", False,
                           f"schema_version={sv} < required {schema_version_min}")
    if expected_doc_id is not None:
        found = sub.get("doc_id")
        if found != expected_doc_id:
            return CheckResult(name, _rel(p), "json_schema", False,
                               f"doc_id={found!r}, expected={expected_doc_id!r}")
    return CheckResult(name, _rel(p), "json_schema", True)


def _check_field_at_least(
    name: str, p: Path, *, json_path: list[str], min_value: int,
) -> CheckResult:
    payload = _read_json(p)
    if payload is None:
        return CheckResult(name, _rel(p), "field_at_least", False, "path missing or unparseable")
    sub: object = payload
    for key in json_path:
        if not isinstance(sub, dict) or key not in sub:
            return CheckResult(name, _rel(p), "field_at_least", False,
                               f"path {'/'.join(json_path)} missing")
        sub = sub[key]
    if not isinstance(sub, (int, float)):
        return CheckResult(name, _rel(p), "field_at_least", False,
                           f"value at {'/'.join(json_path)} is not numeric: {sub!r}")
    if sub < min_value:
        return CheckResult(name, _rel(p), "field_at_least", False,
                           f"value {sub} < required {min_value}")
    return CheckResult(name, _rel(p), "field_at_least", True)


def _check_glob_at_least_one(
    name: str, root: Path, glob: str, *, warning: bool = False,
) -> CheckResult:
    pattern = f"{_rel(root)}/{glob}"
    if not root.exists():
        return CheckResult(name, pattern, "glob_at_least_one", False,
                           "root path missing", warning)
    matches = list(root.glob(glob))
    if not matches:
        return CheckResult(name, pattern, "glob_at_least_one", False,
                           f"no files match {glob}", warning)
    return CheckResult(name, pattern, "glob_at_least_one", True,
                       f"{len(matches)} match(es)", warning)


def _file_sha256(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _check_pdf_sha_match(name: str, pdf_path: Path, expected_sha: str) -> CheckResult:
    if not pdf_path.exists():
        return CheckResult(name, _rel(pdf_path), "pdf_sha_match", False, "PDF not on disk")
    actual = _file_sha256(pdf_path)
    if actual != expected_sha:
        return CheckResult(name, _rel(pdf_path), "pdf_sha_match", False,
                           f"sha mismatch: expected={expected_sha[:12]}…, found={actual[:12] if actual else 'ERR'}…")
    return CheckResult(name, _rel(pdf_path), "pdf_sha_match", True)


def _check_field_equals(
    name: str, p: Path, *, json_path: list[str], expected, validator: str = "field_equals",
) -> CheckResult:
    payload = _read_json(p)
    if payload is None:
        return CheckResult(name, _rel(p), validator, False, "path missing or unparseable")
    sub: object = payload
    for key in json_path:
        if not isinstance(sub, dict) or key not in sub:
            return CheckResult(name, _rel(p), validator, False,
                               f"path {'/'.join(json_path)} missing")
        sub = sub[key]
    if sub != expected:
        return CheckResult(name, _rel(p), validator, False,
                           f"value={sub!r}, expected={expected!r}")
    return CheckResult(name, _rel(p), validator, True)


# ── Discovery: read T02's discovery.json early ───────────────────────────────


def _read_discovery(doc_id: str) -> dict | None:
    p = paths.project_data_dir("T02", doc_id) / "discovery.json"
    return _read_json(p)


# ── Per-project check builders ───────────────────────────────────────────────


def checks_t02(doc_id: str, discovery: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    discovery_path = paths.project_data_dir("T02", doc_id) / "discovery.json"
    out.append(_check_exists("t02_discovery_present", discovery_path))
    if discovery is not None:
        out.append(_check_json_schema("t02_discovery_schema", discovery_path,
                                      schema_version_min=1, expected_doc_id=doc_id))
    pdf_path = paths.project_dir("T02") / "data" / "pdfs" / f"{doc_id}.pdf"
    out.append(_check_non_empty("t02_pdf_on_disk", pdf_path))
    if discovery is not None and discovery.get("pdf_sha256") and pdf_path.exists():
        out.append(_check_pdf_sha_match("t02_pdf_sha_match", pdf_path,
                                        discovery["pdf_sha256"]))
    return out


def checks_t03(doc_id: str, discovery: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    base = paths.project_data_dir("T03", doc_id)
    pdf_meta = base / "pdf_meta.json"
    out.append(_check_exists("t03_pdf_meta_present", pdf_meta))
    out.append(_check_json_schema("t03_pdf_meta_schema", pdf_meta,
                                  schema_version_min=1, expected_doc_id=doc_id))
    if discovery is not None and discovery.get("pdf_sha256"):
        out.append(_check_field_equals(
            "t03_pdf_sha_match_t02", pdf_meta,
            json_path=["pdf_sha256"], expected=discovery["pdf_sha256"],
            validator="matches_t02_sha",
        ))
    out.append(_check_non_empty("t03_raw_text_non_empty", base / "raw_text.txt"))
    out.append(_check_non_empty("t03_article_spans_present", base / "article_spans.json"))
    out.append(_check_glob_at_least_one(
        "t03_at_least_one_config", base / "configs", "*/manifest.json",
    ))
    return out


def checks_t04(doc_id: str, discovery: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    base = paths.project_data_dir("T04", doc_id)
    out.append(_check_glob_at_least_one(
        "t04_at_least_one_config", base / "configs", "*/manifest.json",
    ))
    # Pick the lex-latest config manifest as the per-PDF source of truth (PLAN §B.5).
    configs_dir = base / "configs"
    if configs_dir.exists():
        manifests = sorted(configs_dir.glob("*/manifest.json"))
        if manifests:
            latest = manifests[-1]
            out.append(_check_json_schema(
                "t04_config_doc_id_match", latest,
                schema_version_min=1, expected_doc_id=doc_id,
            ))
            if discovery is not None and discovery.get("pdf_sha256"):
                out.append(_check_field_equals(
                    "t04_config_pdf_sha_match", latest,
                    json_path=["pdf_sha256"], expected=discovery["pdf_sha256"],
                    validator="matches_t02_sha",
                ))
            out.append(_check_field_at_least(
                "t04_linker_version_current", latest,
                json_path=["chunking", "linker_version"], min_value=3,
            ))
    out.append(_check_glob_at_least_one(
        "t04_at_least_one_result_jsonl", base / "results", "*/*.jsonl",
        warning=True,
    ))
    return out


def checks_t05(doc_id: str, discovery: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    base = paths.project_data_dir("T05", doc_id)
    tree_path = base / "tree.json"
    out.append(_check_exists("t05_tree_present", tree_path))
    out.append(_check_field_at_least(
        "t05_tree_builder_version", tree_path,
        json_path=["manifest", "tree_builder_version"], min_value=1,
    ))
    out.append(_check_field_equals(
        "t05_tree_doc_id_match", tree_path,
        json_path=["manifest", "doc_id"], expected=doc_id,
        validator="matches_doc_id",
    ))
    if discovery is not None and discovery.get("pdf_sha256"):
        out.append(_check_field_equals(
            "t05_tree_pdf_sha_match", tree_path,
            json_path=["manifest", "pdf_sha256"], expected=discovery["pdf_sha256"],
            validator="matches_t02_sha",
        ))
    out.append(_check_glob_at_least_one(
        "t05_at_least_one_result", base / "results", "*/*.jsonl", warning=True,
    ))
    return out


def checks_t07(doc_id: str, discovery: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    base = paths.project_data_dir("T07", doc_id)
    qrel = base / "question_relevance.json"
    out.append(_check_non_empty("t07_per_pdf_gt_present", qrel))
    gt_dir = paths.project_dir("T07") / "ground_truth"
    out.append(_check_non_empty("t07_global_gt_train", gt_dir / "bsard_train.json"))
    out.append(_check_non_empty("t07_global_gt_test", gt_dir / "bsard_test.json"))
    out.append(_check_exists("t07_runs_dir_present", gt_dir / "runs", warning=True))
    last_run = base / "eval" / "last_run.json"
    out.append(_check_exists("t07_last_run_present", last_run, warning=True))
    return out


def checks_t08(doc_id: str, discovery: dict | None) -> list[CheckResult]:
    out: list[CheckResult] = []
    bundles_root = paths.project_data_dir("T03", doc_id) / "t08_bundles"
    out.append(_check_glob_at_least_one(
        "t08_arm1_bundles_present", bundles_root, "*/q*.json", warning=True,
    ))
    return out


CHECK_BUILDERS: dict[str, Callable[[str, dict | None], list[CheckResult]]] = {
    "T02": checks_t02,
    "T03": checks_t03,
    "T04": checks_t04,
    "T05": checks_t05,
    "T07": checks_t07,
    "T08": checks_t08,
}


# ── Aggregation ──────────────────────────────────────────────────────────────


def project_status(checks: list[CheckResult]) -> tuple[str, int, int]:
    """Aggregate (status, n_ok_strict, n_strict). 'partial' iff some strict
    checks pass, some fail. Warn-only checks never downgrade status."""
    strict = [c for c in checks if not c.warning]
    n_strict = len(strict)
    n_ok = sum(1 for c in strict if c.ok)
    if n_strict == 0:
        # All checks for this project are warn-only -> informational
        return "complete", 0, 0
    if n_ok == n_strict:
        return "complete", n_ok, n_strict
    if n_ok == 0:
        return "missing", 0, n_strict
    return "partial", n_ok, n_strict


def overall_status(per_project: dict[str, str]) -> str:
    relevant = [per_project[code] for code in PROJECTS_IN_OVERALL if code in per_project]
    if not relevant:
        return "missing"
    if all(s == "complete" for s in relevant):
        return "complete"
    if any(s == "complete" for s in relevant):
        return "partial"
    if any(s == "partial" for s in relevant):
        return "partial"
    return "missing"


# ── Registry build + write ───────────────────────────────────────────────────


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_registry(doc_id: str, project_filter: list[str] | None = None) -> dict:
    discovery = _read_discovery(doc_id)
    in_curated = doc_id in paths.selected_pdf_doc_ids()

    selected = project_filter or list(ALL_PROJECTS)
    projects: dict[str, dict] = {}
    per_project_status: dict[str, str] = {}

    for code in selected:
        if code not in CHECK_BUILDERS:
            logger.warning("Unknown project code in filter: %s", code)
            continue
        checks = CHECK_BUILDERS[code](doc_id, discovery)
        status, n_ok, n_strict = project_status(checks)
        projects[code] = {
            "status": status,
            "n_checks": n_strict,
            "n_ok": n_ok,
            "checks": [c.to_dict() for c in checks],
        }
        per_project_status[code] = status

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "last_checked_at": _now_utc(),
        "rq2_root": paths.RQ2_ROOT.as_posix(),
        "in_curated_set": in_curated,
        "overall": overall_status(per_project_status),
        "projects": projects,
    }


def write_registry(registry: dict) -> Path:
    out_path = _status_dir() / f"{registry['doc_id']}.json"
    _assert_inside_status_dir(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def write_index(per_pdf: list[dict]) -> Path:
    out_path = _status_dir() / "index.json"
    _assert_inside_status_dir(out_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "computed_at": _now_utc(),
        "doc_ids": [r["doc_id"] for r in per_pdf],
        "summary": [
            {
                "doc_id": r["doc_id"],
                "overall": r["overall"],
                "per_project": {code: r["projects"][code]["status"]
                                for code in r["projects"]},
            }
            for r in per_pdf
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


# ── Rendering ────────────────────────────────────────────────────────────────


_STATUS_GLYPH = {"complete": "OK", "partial": "PARTIAL", "missing": "MISSING"}


def render_table(registry: dict) -> str:
    rows = []
    rows.append(f"doc_id: {registry['doc_id']}    overall: {registry['overall']}    in_curated_set: {registry['in_curated_set']}")
    rows.append("")
    rows.append(f"{'Project':<6}  {'Status':<8}  {'OK/N':>7}  {'Worst-failing check':<48}")
    rows.append("-" * 80)
    for code in ALL_PROJECTS:
        if code not in registry["projects"]:
            continue
        proj = registry["projects"][code]
        worst = next(
            (c for c in proj["checks"] if not c["ok"] and not c.get("warning")),
            None,
        )
        worst_label = (
            f"{worst['name']}: {worst.get('detail') or '-'}" if worst else "-"
        )
        if len(worst_label) > 46:
            worst_label = worst_label[:45] + "…"
        rows.append(
            f"{code:<6}  {_STATUS_GLYPH[proj['status']]:<8}  "
            f"{proj['n_ok']}/{proj['n_checks']:<5}  {worst_label:<48}"
        )
    return "\n".join(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(prog="orchestrator.status",
                                     description=__doc__.split("\n", 1)[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-id", help="Single PDF stem (e.g. 1804_03_21_1804032150)")
    group.add_argument("--all", action="store_true",
                       help="Loop over RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json")
    parser.add_argument("--project", action="append", choices=ALL_PROJECTS,
                       help="Restrict to one or more project codes")
    parser.add_argument("--json", action="store_true",
                       help="Emit registry JSON to stdout (does not write to disk)")
    parser.add_argument("--strict", action="store_true",
                       help="Exit non-zero if overall != 'complete'")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_filter = list(args.project) if args.project else None

    doc_ids = paths.selected_pdf_doc_ids() if args.all else [args.doc_id]

    # --project narrows the view for exploration; never persist a partial
    # view (it would clobber the full registry from a prior --all run).
    persist = not args.json and project_filter is None

    per_pdf_registries: list[dict] = []
    for doc_id in doc_ids:
        registry = build_registry(doc_id, project_filter=project_filter)
        per_pdf_registries.append(registry)
        if args.json:
            print(json.dumps(registry, indent=2, ensure_ascii=False))
        else:
            print(render_table(registry))
            print()
            if persist:
                out_path = write_registry(registry)
                print(f"Wrote {_rel(out_path)}")
            else:
                print("(filtered view — registry on disk left unchanged)")
            print()

    if args.all and persist:
        index_path = write_index(per_pdf_registries)
        print(f"Wrote {_rel(index_path)}")

    if args.strict:
        worst = next(
            (r["overall"] for r in per_pdf_registries if r["overall"] != "complete"),
            None,
        )
        if worst is not None:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
