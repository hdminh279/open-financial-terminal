"""Financial Metric Registry and Metadata Resolver (SSOT)."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import yaml


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable domain representation of a standardized financial metric."""

    key: str
    display_name: str
    category: str
    classification: str
    period_type: str
    default_unit: str
    storage_layer: str
    is_dimensional: bool
    description: str
    synonyms: List[str] = field(default_factory=list)


class MetricRegistry:
    """In-memory registry and fast resolver for taxonomy metrics."""

    DEFAULT_TAXONOMY_PATH = "configs/xbrl_taxonomy_map.yaml"

    def __init__(self, config_path: str = DEFAULT_TAXONOMY_PATH) -> None:
        self.config_path = config_path
        self._metrics: Dict[str, MetricDefinition] = {}
        self._synonym_index: Dict[str, str] = {}
        self._load_and_index()

    def _load_and_index(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Taxonomy config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        metrics_raw = data.get("metrics", {})
        for metric_key, info in metrics_raw.items():
            if not isinstance(info, dict):
                continue

            synonyms = [s.strip().lower() for s in info.get("synonyms", []) if isinstance(s, str)]
            definition = MetricDefinition(
                key=metric_key,
                display_name=info.get("display_name", metric_key),
                category=info.get("category", "unclassified"),
                classification=info.get("classification", "gaap"),
                period_type=info.get("period_type", "duration"),
                default_unit=info.get("default_unit", "USD"),
                storage_layer=info.get("storage_layer", "sql_gold"),
                is_dimensional=bool(info.get("is_dimensional", False)),
                description=info.get("description", ""),
                synonyms=synonyms,
            )
            self._metrics[metric_key] = definition

            # Index canonical key and display name
            self._synonym_index[metric_key.lower()] = metric_key
            self._synonym_index[definition.display_name.lower()] = metric_key

            # Index all synonyms
            for syn in synonyms:
                self._synonym_index[syn] = metric_key

    def get(self, metric_key: str) -> Optional[MetricDefinition]:
        """Retrieve metric definition by canonical key."""
        return self._metrics.get(metric_key.strip().lower())

    def resolve(self, query_term: str) -> Optional[MetricDefinition]:
        """Resolve a bilingual synonym, display name, or canonical key to its definition."""
        if not query_term or not isinstance(query_term, str):
            return None
        norm_term = query_term.strip().lower()
        canonical_key = self._synonym_index.get(norm_term)
        return self._metrics.get(canonical_key) if canonical_key else None

    def is_sql_metric(self, metric_key: str) -> bool:
        """Check if metric is queryable via DuckDB SQL Gold layer."""
        metric = self.get(metric_key)
        return metric.storage_layer == "sql_gold" if metric else False

    def is_rag_metric(self, metric_key: str) -> bool:
        """Check if metric requires textual extraction via RAG layer."""
        metric = self.get(metric_key)
        return metric.storage_layer == "rag_textual" if metric else False

    def get_default_unit(self, metric_key: str) -> str:
        """Return standardized unit ('USD', '%', 'ratio', 'USD/shares')."""
        metric = self.get(metric_key)
        return metric.default_unit if metric else "USD"

    def list_metrics(
        self,
        category: Optional[str] = None,
        storage_layer: Optional[str] = None,
    ) -> List[str]:
        """List canonical keys filtered optionally by category or storage layer."""
        results = []
        for key, m in self._metrics.items():
            if category and m.category != category:
                continue
            if storage_layer and m.storage_layer != storage_layer:
                continue
            results.append(key)
        return results
