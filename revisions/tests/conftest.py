"""
Fixtures for the collector tests. Nothing here imports torch at module level, so the CbD tests still
run without it.
"""

import pytest

from revisions.tests.collector_data import MIXED_CASE_PAIR, PAIRS, all_sentences


@pytest.fixture
def data_dir(tmp_path):
    """Pairs TSVs, the per-sentence null input and the referent-only null input, in tmp_path."""
    from revisions.build_refnull_inputs import build_refnull_inputs

    PAIRS.to_csv(tmp_path / "pairs.tsv", sep="\t", index=False)
    MIXED_CASE_PAIR.to_csv(tmp_path / "mixed.tsv", sep="\t", index=False)
    all_sentences(PAIRS).to_csv(tmp_path / "all_sentences.csv", index=False)
    build_refnull_inputs(PAIRS).to_csv(tmp_path / "refnull.csv", index=False)
    return tmp_path


@pytest.fixture
def use_fake(monkeypatch):
    """Swap collect_sequential's ModelProbs for a fake with the given answer policy; returns the fake class."""
    from revisions.tests.fake_model import fake_model_probs
    from winogender_contextuality.modeling import collect_sequential as cs

    def _use(policy=None):
        fake = fake_model_probs(policy) if policy else fake_model_probs()
        monkeypatch.setattr(cs, "ModelProbs", fake)
        return fake

    return _use
