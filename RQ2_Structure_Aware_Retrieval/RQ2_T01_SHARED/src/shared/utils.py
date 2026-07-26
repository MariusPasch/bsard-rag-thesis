"""Logging, timing, and text utilities."""
from __future__ import annotations

import logging
import random
import re
import time
from functools import wraps

import numpy as np

# French legal abbreviations that end with a period but don't terminate a sentence.
_ABBREVS = {
    "art", "al", "cf", "ibid", "op", "cit", "loc", "no", "n°",
    "vol", "p", "pp", "fig", "tab", "dr", "me", "mme", "mm",
    "jr", "sr", "etc", "env", "approx", "sect", "par", "§",
}

_SENT_BOUNDARY = re.compile(r"\.\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]")
_ABBREVS_LOWER = frozenset(a.lower() for a in _ABBREVS)


def first_sentence(text: str) -> str:
    """Extract first sentence from text, respecting French legal abbreviations."""
    text = text.strip()
    for m in _SENT_BOUNDARY.finditer(text):
        before = text[: m.start()]
        last_word = re.split(r"\s+", before)[-1].rstrip(".").lower()
        if last_word not in _ABBREVS_LOWER:
            return text[: m.start() + 1]
    return text


def timer(func):
    """Decorator that logs execution time of the wrapped function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("%s completed in %.1f ms", func.__qualname__, elapsed_ms)
        return result
    return wrapper


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure and return a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def set_seeds(seed: int = 42) -> None:
    """Fix random seeds for numpy, torch, and the stdlib random module."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
