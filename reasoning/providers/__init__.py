"""VLM provider registry.

Importing the registry must not trigger model loads. Concrete providers
defer their heavy imports to first ``analyze_evidence`` call.
"""
from .base import (
    EvidenceManifest,
    ProviderHealth,
    VLMProvider,
    VLMResult,
    get_provider,
    list_providers,
    register_provider,
)

# Side-effect registers built-in providers.
from . import gemma  # noqa: F401
from . import qwen3_vl  # noqa: F401

__all__ = [
    "EvidenceManifest",
    "ProviderHealth",
    "VLMProvider",
    "VLMResult",
    "get_provider",
    "list_providers",
    "register_provider",
]
