"""
Off-GPU tests of the single-blank collectors (primed, generic null, referent-only null), run for real
against a fake model, and of the referent-only prime builder.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from revisions.build_refnull_inputs import build_refnull_inputs, mention_order
from revisions.tests.collector_data import PAIRS
from revisions.tests.fake_model import PRONOUN_IDS
from winogender_contextuality.modeling import collect_sequential as cs
from winogender_contextuality.utils import load_ndjson


def test_refnull_primes_follow_sentence_order_and_name_both_referents():
    out = build_refnull_inputs(PAIRS)
    assert list(out.template) == [PAIRS.template_1[0], PAIRS.template_2[0], PAIRS.template_1[1], PAIRS.template_2[1]]
    assert list(out.prime[:2]) == ["The technician and the customer were both present."] * 2
    assert list(out.prime[2:]) == ["The customer and the technician were both present."] * 2
    assert list(out.columns) == ["template", "differences", "referent", "case", "prime"]


def test_mention_order_rejects_a_referent_the_sentence_does_not_mention():
    with pytest.raises(ValueError):
        mention_order("The nurse told the patient that BLANK left.", ["nurse", "doctor"])


def test_null_run_defaults_are_unchanged_and_logits_cover_both_options(data_dir, use_fake):
    fake = use_fake()
    out = data_dir / "null.ndjson"
    cs.generate_one_null_context("gpu", "fake/model", 0.5, n_runs=2, input_dir=data_dir,
                                 input_file="all_sentences.csv", output_file=out)
    records = load_ndjson(out)

    assert len(records) == 4 * 3 * 2 * 2  # sentences x generic primes x option orders x runs
    assert {r["context"]["sent_order"][0] for r in records} == {"null_0", "null_1", "null_2"}
    assert all(r["context"]["pnoun_order"][0] == "null" for r in records)
    # Logits over the two options, not over the letters of 'she' (the old pronouns[1] bug)
    for r in records:
        canonical = r["context"]["pronouns_2"] if r["context"]["pnoun_order"][1] == 0 else r["context"]["pronouns_2"][::-1]
        assert list(np.ravel(r["logits"])) == [PRONOUN_IDS[p] for p in canonical]
    assert fake.instances[0].quantized is True


def test_refnull_run_uses_each_rows_prime(data_dir, use_fake):
    fake = use_fake()
    out = data_dir / "refnull.ndjson"
    cs.generate_one_null_context("gpu", "fake/model", 0.5, n_runs=2, input_dir=data_dir, input_file="refnull.csv",
                                 output_file=out, prime_column="prime", tag="refnull", quantized=False)
    records = load_ndjson(out)
    primes = build_refnull_inputs(PAIRS).prime

    assert len(records) == 4 * 1 * 2 * 2
    assert {r["context"]["sent_order"][0] for r in records} == {"refnull_0"}
    assert all(r["context"]["sentence_1"] == primes[r["index"]] for r in records)
    assert all(r["context"]["pnoun_order"][0] == "refnull" for r in records)
    # The prime really is in the prompt, right before the free sentence
    for prompt, r in zip(fake.prompts, records):
        assert f"*{r['context']['sentence_1']} {r['context']['sentence_2']}*" in prompt[1]["content"]
    assert fake.instances[0].quantized is False


def test_primed_logits_use_the_free_sentences_options(data_dir, use_fake):
    use_fake()
    out = data_dir / "primed.ndjson"
    cs.generate_one_pronoun("gpu", "fake/model", 0.5, n_runs=1, input_dir=data_dir, input_file="mixed.tsv",
                            output_file=out)
    records = load_ndjson(out)

    assert {tuple(r["context"]["sent_order"]) for r in records} == {(0, 1), (1, 0)}
    for r in records:
        assert list(np.ravel(r["logits"])) == [PRONOUN_IDS[p] for p in r["context"]["pronouns_2"]]
