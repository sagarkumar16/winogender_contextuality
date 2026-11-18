"""
Small inputs for the collector tests: two real WinoPron pairs, and one pair whose sentences differ in
grammatical case.
"""

import pandas as pd

# Two real WinoPron pairs (nominative and possessive)
PAIRS = pd.DataFrame([
    {"template_1": "The technician told the customer that BLANK could pay with cash.",
     "differences_1": "['he', 'she']", "case_1": "$NOM_PRONOUN", "referent_1": "customer",
     "template_2": "The technician told the customer that BLANK had completed the repair.",
     "differences_2": "['he', 'she']", "case_2": "$NOM_PRONOUN", "referent_2": "technician"},
    {"template_1": "The customer told the technician that BLANK fridge was no longer working.",
     "differences_1": "['his', 'her']", "case_1": "$POSS_PRONOUN", "referent_1": "customer",
     "template_2": "The customer told the technician that BLANK charges were too high.",
     "differences_2": "['his', 'her']", "case_2": "$POSS_PRONOUN", "referent_2": "technician"},
])

# A pair whose sentences differ in case -- none in WinoPron do, but the collectors must not rely on that
MIXED_CASE_PAIR = pd.DataFrame([
    {"template_1": "The nurse told the patient that BLANK shift was over.",
     "differences_1": "['his', 'her']", "case_1": "$POSS_PRONOUN", "referent_1": "nurse",
     "template_2": "The nurse told the patient that BLANK would be discharged soon.",
     "differences_2": "['he', 'she']", "case_2": "$NOM_PRONOUN", "referent_2": "patient"},
])


def all_sentences(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence, template_1 then template_2 -- as the notebook builds all_sentences_wp."""
    rows = []
    for idx in pairs.index:
        for n in [1, 2]:
            rows.append({"template": pairs[f"template_{n}"][idx], "differences": pairs[f"differences_{n}"][idx],
                         "referent": pairs[f"referent_{n}"][idx], "case": pairs[f"case_{n}"][idx]})
    return pd.DataFrame(rows)
