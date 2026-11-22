"""Externalized bilingual response templates and query language detection.

Loads localized user-facing text and multilingual routing phrase lists from
configs/response_templates.yaml so all Python modules stay strictly English.
"""

import os
import re
from functools import lru_cache
from typing import Any, Literal

import yaml

DEFAULT_TEMPLATES_PATH = "configs/response_templates.yaml"

# Unicode range covering standard Vietnamese diacritical characters
_VIETNAMESE_DIACRITIC_RE = re.compile(
    r"[\u00C0-\u00C3\u00C8-\u00CA\u00CC-\u00CD\u00D2-\u00D5\u00D9-\u00DA\u00DD"
    r"\u00E0-\u00E3\u00E8-\u00EA\u00EC-\u00ED\u00F2-\u00F5\u00F9-\u00FA\u00FD"
    r"\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01A0-\u01B0\u1EA0-\u1EF9]"
)


@lru_cache(maxsize=4)
def load_response_config(config_path: str = DEFAULT_TEMPLATES_PATH) -> dict[str, Any]:
    """Load and cache response template configuration YAML."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def detect_query_language(query: str, config_path: str = DEFAULT_TEMPLATES_PATH) -> Literal["vi", "en"]:
    """Detect whether a user query is written in Vietnamese or English."""
    text = (query or "").strip()
    if not text:
        return "en"
    if _VIETNAMESE_DIACRITIC_RE.search(text):
        return "vi"
    lowered = text.lower()
    cfg = load_response_config(config_path)
    vi_keywords = (cfg.get("language_detection") or {}).get("vietnamese_keywords") or []
    if any(kw in lowered for kw in vi_keywords):
        return "vi"
    return "en"


def get_routing_markers(
    group: str,
    config_path: str = DEFAULT_TEMPLATES_PATH,
) -> tuple[str, ...]:
    """Retrieve a named tuple of lowercase routing marker phrases from config."""
    cfg = load_response_config(config_path)
    markers = (cfg.get("routing_markers") or {}).get(group) or []
    return tuple(str(m).lower() for m in markers if isinstance(m, str) and m)


def render_template(
    template_key: str,
    lang: Literal["vi", "en"] = "en",
    config_path: str = DEFAULT_TEMPLATES_PATH,
    **kwargs: Any,
) -> str:
    """Render a localized user-facing response template with named parameters."""
    cfg = load_response_config(config_path)
    templates = cfg.get("templates") or {}
    entry = templates.get(template_key) or {}
    raw_template = entry.get(lang) or entry.get("en") or ""
    if not raw_template:
        return ""
    return raw_template.format(**kwargs)
