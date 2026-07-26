"""
Minimal setup.py for editable install (pip install -e .).
Allows RQ1 and other projects to import bsard_evaluation as a package.
"""
from setuptools import setup, find_packages

setup(
    name="bsard-evaluation",
    version="0.1.0",
    description="Canonical evaluation metrics for the BSARD RAG thesis (RQ3)",
    author="Marios Paschalidis",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7", "seaborn>=0.12"],
        "tier3": [
            "ragas>=0.1",
            "ragchecker>=0.1",
            "deepeval>=0.20",
            "ares-ai>=0.1",
            "datasets>=2.14",
            "openai>=1.0",
        ],
        "dev": ["pytest>=7.0", "jupyter>=1.0", "ipykernel>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "bsard-eval=scripts.evaluate:main",
            "bsard-compare=scripts.compare:main",
        ],
    },
)
