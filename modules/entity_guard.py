"""
entity_guard.py
───────────────
Protects known named entities (people, places, org names) from being
distorted during Odia→English translation.

Flow:
  1. protect(text)  → replaces Odia entity with <<E_0>>, <<E_1>>, ...
  2. (translation happens)
  3. restore(text, mapping) → puts canonical English form back

Example:
  Input : "ଯୀଶୁ ଜଗତକୁ ଭଲ ପାଉଥିଲେ"
  After protect: "<<E_0>> ଜଗତକୁ ଭଲ ପାଉଥିଲେ"  |  mapping = {"<<E_0>>": "Jesus"}
  After translate: "<<E_0>> loved the world"
  After restore: "Jesus loved the world"
"""

import json
import re
from config import ENTITY_DICT_PATH


def load_entity_dict(path: str = ENTITY_DICT_PATH) -> dict:
    """Load the Odia→English entity dictionary from JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def protect_entities(text: str, entity_dict: dict) -> tuple[str, dict]:
    """
    Replace known Odia entities with safe placeholders.

    Returns:
        protected_text : text with placeholders inserted
        mapping        : {placeholder: canonical_english_form}
    """
    mapping = {}
    counter = 0

    # Sort by length descending so longer matches take priority
    # e.g. "ଯୀଶୁ ଖ୍ରୀଷ୍ଟ" matched before "ଯୀଶୁ"
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
    Also handles cases where the translator may have modified the
    placeholder slightly (e.g. added spaces inside << >>).
    """
    restored = text
    for placeholder, english_name in mapping.items():
        # Exact match first
        if placeholder in restored:
            restored = restored.replace(placeholder, english_name)
        else:
            # Fuzzy match: handle <<E_ 0>> or << E_0 >> style artifacts
            idx = placeholder.replace("_", "").replace(" ", "")
            pattern = r'<<\s*E\s*_?\s*' + re.escape(
                placeholder.strip("<<>>").replace("E_", "")
            ) + r'\s*>>'
            restored = re.sub(pattern, english_name, restored)

    return restored