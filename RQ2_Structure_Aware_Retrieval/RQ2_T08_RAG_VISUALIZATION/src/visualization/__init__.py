"""Local Streamlit viewer for inspecting chunks, articles, and retrieval results on the original PDF page layout."""

from visualization.launcher import launch
from visualization.region_adapter import HighlightRegion, HighlightCategory

__all__ = ["launch", "HighlightRegion", "HighlightCategory"]
