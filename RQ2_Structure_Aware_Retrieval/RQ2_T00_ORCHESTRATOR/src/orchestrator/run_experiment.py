"""Main entry point for the RQ2 pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    logs_dir = Path(config["output"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "run_experiment.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ2 pipeline — Belgian statutory article retrieval")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--methods", nargs="+", default=["arm1", "2A", "2B"],
        choices=["arm1", "2A", "2B"],
        help="Which retrieval arms to run",
    )
    parser.add_argument("--pdf", nargs="+", help="Run only for these PDF filenames")
    parser.add_argument("--variants", nargs="+", help="Arm 2A variants to run")
    parser.add_argument("--skip-indexing", action="store_true", help="Reuse existing indices")
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild all indices even if they exist")
    parser.add_argument("--eval-only", action="store_true", help="Skip retrieval, run eval on saved results")
    return parser.parse_args()


def load_models(config: dict):
    """Load embedding model once at startup."""
    from shared.embeddings import EmbeddingModel

    logger.info("Loading embedding model: %s", config["models"]["embedding_model"])
    embedding_model = EmbeddingModel(
        model_name=config["models"]["embedding_model"],
        fallback_model=config["models"].get("embedding_model_fallback"),
        max_tokens=config["models"]["embedding_max_tokens"],
    )

    return embedding_model


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to this file's location
        config_path = Path(__file__).parent.parent.parent / args.config
    config = load_config(str(config_path))

    setup_logging(config)
    logger.info("Config loaded from %s", config_path)

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        _run_eval_only(config, results_dir)
        return

    # Load shared models once — reused across all documents and arms
    embedding_model = load_models(config)

    # Load documents
    from data_loader import load_documents
    document_bundles = load_documents(config)
    logger.info("Loaded %d document bundle(s)", len(document_bundles))

    db_path = Path(config["data"]["bsard_db"])
    index_dir = Path(config["data"]["index_dir"])
    force_reindex = args.force_reindex and not args.skip_indexing

    all_results: dict = {}

    for bundle in document_bundles:
        if args.pdf and bundle.pdf_filename not in args.pdf:
            continue

        logger.info("Processing document: %s", bundle.pdf_filename)

        # Arm 1 — naive chunking
        if "arm1" in args.methods:
            from arm1_naive.retriever import run_arm1
            logger.info("Running Arm 1 for %s", bundle.pdf_filename)
            results_arm1 = run_arm1(
                bundle=bundle,
                embedding_model=embedding_model,
                tokenizer=embedding_model.tokenizer,
                db_path=db_path,
                index_dir=index_dir / "arm1",
                config=config,
                force_reindex=force_reindex,
            )
            all_results[f"{bundle.document_id}_arm1"] = results_arm1

        # Arm 2 methods — require Azure extraction
        if bundle.has_azure_extraction:

            if "2A" in args.methods:
                from arm2_metadata import run_metadata_filtering
                variants = args.variants or config["arm2"]["metadata_filtering"]["variants"]
                for variant in variants:
                    logger.info("Running Arm 2A variant=%s for %s", variant, bundle.pdf_filename)
                    results_2a = run_metadata_filtering(bundle, config, variant=variant)
                    all_results[f"{bundle.document_id}_2A_{variant}"] = results_2a

            if "2B" in args.methods:
                from arm2_pageindex import run_pageindex
                logger.info("Running Arm 2B for %s", bundle.pdf_filename)
                results_2b = run_pageindex(bundle, config)
                all_results[f"{bundle.document_id}_2B"] = results_2b

        else:
            logger.warning("No Azure extraction for %s — skipping Arm 2", bundle.pdf_filename)

    # Evaluation
    from evaluation import evaluate, generate_comparison, load_ground_truth, ground_truth_exists
    ground_truth = load_ground_truth(config) if ground_truth_exists(config) else None
    eval_report = evaluate(all_results, ground_truth, config)

    _save_results(all_results, eval_report, config, results_dir)
    generate_comparison(eval_report, config)
    logger.info("Pipeline complete. Results in %s", results_dir)


def _run_eval_only(config: dict, results_dir: Path) -> None:
    import json
    from evaluation import evaluate, generate_comparison, load_ground_truth, ground_truth_exists

    saved = {}
    for result_file in results_dir.rglob("*.json"):
        key = result_file.stem
        with open(result_file, encoding="utf-8") as f:
            saved[key] = json.load(f)

    ground_truth = load_ground_truth(config) if ground_truth_exists(config) else None
    eval_report = evaluate(saved, ground_truth, config)
    generate_comparison(eval_report, config)
    logger.info("Eval-only complete.")


def _save_results(all_results: dict, eval_report, config: dict, results_dir: Path) -> None:
    import json

    if config["output"].get("save_retrieval_lists"):
        for key, results in all_results.items():
            out_path = results_dir / f"{key}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    [r.to_dict() if hasattr(r, "to_dict") else r for r in results],
                    f, indent=2, ensure_ascii=False,
                )

    eval_path = results_dir / "eval_report.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            eval_report.to_dict() if hasattr(eval_report, "to_dict") else eval_report,
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
