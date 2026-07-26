"""Arm 2A: AzureDI-metadata-aware retrieval (enriched embeddings + filter/boost).

Reuses the T03 hybrid stack (mE5-instruct + FAISS, BM25, RRF) but operates
on AzureDI tree nodes (or aggregated articles) and exploits the AzureDI
metadata at index time and at query time.
"""
