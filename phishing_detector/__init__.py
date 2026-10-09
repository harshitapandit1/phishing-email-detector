"""Phishing email detector package.

Only the lightweight inference dependencies are imported eagerly so
that ``server.py`` and ``infer.py`` work without the heavy training
dependencies (``datasets``, ``huggingface_hub``, etc.).
"""

from .features import FeaturePipeline, StructuralFeatureExtractor, TfidfFeatureExtractor

__all__ = [
    "FeaturePipeline",
    "StructuralFeatureExtractor",
    "TfidfFeatureExtractor",
]
