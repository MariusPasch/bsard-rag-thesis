"""Arm 2B: ToC-tree construction and LLM-guided hierarchical navigation (vectorless)."""

from .navigator import NavigationState, NavigatorConfig, navigate
from .pipeline import load_subset_results, run_subset
from .retriever import DEFAULT_METHOD_NAME, run_arm2b, state_to_retrieval_result
from .tree_builder import (
    ToCNode,
    TREE_BUILDER_VERSION,
    build_law_tree,
    compose_corpus_tree,
    find_node,
    is_cache_fresh,
    iter_leaves,
    load_law_tree,
    save_law_tree,
    tree_stats,
)

__all__ = [
    "DEFAULT_METHOD_NAME",
    "NavigationState",
    "NavigatorConfig",
    "ToCNode",
    "TREE_BUILDER_VERSION",
    "build_law_tree",
    "compose_corpus_tree",
    "find_node",
    "is_cache_fresh",
    "iter_leaves",
    "load_law_tree",
    "load_subset_results",
    "navigate",
    "run_arm2b",
    "run_subset",
    "save_law_tree",
    "state_to_retrieval_result",
    "tree_stats",
]
