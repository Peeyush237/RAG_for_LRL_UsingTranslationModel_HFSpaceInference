"""
entity_guard.py
───────────────
Entity protection for translation — copied from original with
path reference updated to use backend config.
"""

import json
import re
import os
from app.config import settings


def load_entity_dict(path: str = "") -> dict:
    """Load the Odia→English entity dictionary from JSON."""
    if not path:
        path = settings.ENTITY_DICT_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def protect_entities(text: str, entity_dict: dict) -> tuple[str, dict]:
    """
    Replace known Odia entities with safe placeholders.

    Returns:
        protected_text: text with placeholders inserted
        mapping:        {placeholder: canonical_english_form}
    """
    mapping = {}
    counter = 0

    # Sort by length descending so longer matches take priority
    sorted_entities = sorted(entity_dict.keys(), key=len, reverse=True)

    protected = text
    for odia_entity in sorted_entities:
        if odia_entity in protected:
            placeholder = f"<<E_{counter}>>"
            protected = protected.replace(odia_entity, placeholder)
            mapping[placeholder] = entity_dict[odia_entity]
            counter += 1

    return protected, mapping


def restore_entities(text: str, mapping: dict) -> str:
    """
    Replace placeholders back with canonical English entity names.
    Handles cases where translator may have modified the placeholder.
    """
    restored = text
    for placeholder, english_name in mapping.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, english_name)
        else:
            # Fuzzy match for mangled placeholders
            idx_num = placeholder.strip("<<>>").replace("E_", "")
            pattern = r'<<\s*E\s*_?\s*' + re.escape(idx_num) + r'\s*>>'
            restored = re.sub(pattern, english_name, restored)

    return restored
