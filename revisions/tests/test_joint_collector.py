"""
Off-GPU tests of the joint-measurement collector, run for real against a fake model, and of scoring its
output with the package's ΔC estimator.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from revisions.cbd import context_stats_from_table, delta_c_rank2
from revisions.tests.collector_data import PAIRS
from revisions.tests.fake_model import FEMALE
from winogender_contextuality.modeling import collect_sequential as cs
from winogender_contextuality.modeling.contextuality import calculate_sentence_nc_fraction, sentence_order_results
from winogender_contextuality.utils import get_index, load_ndjson


def test_joint_records_map_blanks_to_their_sentences(data_dir, use_fake):
    # template_1's pronoun is always female and template_2's always male, whichever comes first
    def by_sentence(passage, options, rng):
        first_is_template_1 = any(t.replace("BLANK", "BLANK1") in passage for t in PAIRS.template_1)
        female, male = [next(o for o in opts if o in FEMALE) for opts in options], \
                       [next(o for o in opts if o not in FEMALE) for opts in options]
        return [female[0], male[1]] if first_is_template_1 else [male[0], female[1]]

    use_fake(by_sentence)
    out = data_dir / "joint.ndjson"
    cs.generate_two_pronouns("gpu", "fake/model", 0.5, n_runs=3, input_dir=data_dir, input_file="pairs.tsv",
                             output_file=out)
    records = load_ndjson(out)

    assert len(records) == 2 * 2 * 2 * 3  # pairs x passage orders x matched option orders x runs
    assert {tuple(r["context"]["pnoun_order"]) for r in records} == {(0, 0), (1, 1)}
    assert all(r["context"]["case_1"] and r["context"]["case_2"] for r in records)
    assert len(get_index(0, records)) == 12  # nothing lost to parsing

    for idx in range(2):
        female = PAIRS.differences_1[idx].split("'")[3]
        for k in (0, 1):
            d = sentence_order_results(idx, records, pnoun_order=[k, k])
            for order in ("forward", "reverse"):
                assert set(d[order]["pnoun_1"]) == {female}       # template_1
                assert female not in set(d[order]["pnoun_2"])     # template_2


def test_joint_option_orders_reach_the_prompt(data_dir, use_fake):
    fake = use_fake()  # always answers the option presented first
    out = data_dir / "joint.ndjson"
    cs.generate_two_pronouns("gpu", "fake/model", 0.5, n_runs=1, input_dir=data_dir, input_file="pairs.tsv",
                             output_file=out, option_orders="all")
    records = load_ndjson(out)

    assert {tuple(r["context"]["pnoun_order"]) for r in records} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for r in records:
        i, j = r["context"]["pnoun_order"]
        assert (r["measurement"]["BLANK1"] in FEMALE) == (i == 1)
        assert (r["measurement"]["BLANK2"] in FEMALE) == (j == 1)
    assert len(fake.prompts) == len(records)

    with pytest.raises(ValueError):
        cs.generate_two_pronouns("gpu", "fake/model", 0.5, input_dir=data_dir, input_file="pairs.tsv",
                                 output_file=out, option_orders="mixed")


def test_joint_without_assistant_prefill(data_dir, use_fake):
    fake = use_fake()
    out = data_dir / "joint.ndjson"
    cs.generate_two_pronouns("gpu", "fake/model", 0.5, n_runs=1, input_dir=data_dir, input_file="pairs.tsv",
                             output_file=out, assistant=False)
    records = load_ndjson(out)

    assert all(p[-1]["role"] == "user" for p in fake.prompts)
    assert all(r["measurement"]["BLANK1"] != "None" for r in records)


def test_joint_delta_c_matches_the_independent_cbd_implementation(data_dir, use_fake):
    # Correlated, order-dependent answers, so every term of delta_c is non-trivial
    def correlated(passage, options, rng):
        fem = [next(o for o in opts if o in FEMALE) for opts in options]
        mal = [next(o for o in opts if o not in FEMALE) for opts in options]
        p1 = 0.7 if any(t.replace("BLANK", "BLANK1") in passage for t in PAIRS.template_1) else 0.4
        a1 = rng.random() < p1
        a2 = rng.random() < (0.8 if a1 else 0.3)
        return [fem[0] if a1 else mal[0], fem[1] if a2 else mal[1]]

    use_fake(correlated)
    out = data_dir / "joint.ndjson"
    cs.generate_two_pronouns("gpu", "fake/model", 0.5, n_runs=60, input_dir=data_dir, input_file="pairs.tsv",
                             output_file=out)
    records = load_ndjson(out)

    for idx in range(2):
        for k in (0, 1):
            package = calculate_sentence_nc_fraction(sentence_order_results(idx, records, pnoun_order=[k, k]))

            # Independent: forward table is [A female, B female]; reverse is [B female, A female],
            # since BLANK1 is the sentence shown first (cbd.delta_c_rank2's convention)
            tables = {}
            for order in ([0, 1], [1, 0]):
                t = np.zeros((2, 2))
                for r in records:
                    if r["index"] == idx and r["context"]["sent_order"] == order and r["context"]["pnoun_order"] == [k, k]:
                        t[int(r["measurement"]["BLANK1"] in FEMALE), int(r["measurement"]["BLANK2"] in FEMALE)] += 1
                tables[tuple(order)] = t
            independent = delta_c_rank2(context_stats_from_table(tables[(0, 1)]),
                                        context_stats_from_table(tables[(1, 0)]))
            assert package == pytest.approx(independent, abs=1e-12)


@pytest.mark.parametrize("text, expected", [
    ("{'BLANK1':'she', 'BLANK2':'he'}", {"BLANK1": "she", "BLANK2": "he"}),
    ("{'BLANK1':'her', 'BLANK2': 'his'}\nThe passage says...", {"BLANK1": "her", "BLANK2": "his"}),
    ("model\n{'BLANK1': 'him', 'BLANK2': 'her'}", {"BLANK1": "him", "BLANK2": "her"}),
    ("{'BLANK1':'she', 'BLANK2':'h", {"BLANK1": "None", "BLANK2": "None"}),
    ("{'BLANK1':'she'}", {"BLANK1": "None", "BLANK2": "None"}),
    ("I cannot answer that.", {"BLANK1": "None", "BLANK2": "None"}),
])
def test_parse_joint_completion(text, expected):
    assert cs.parse_joint_completion(text) == expected


def test_get_index_drops_unparsed_single_and_joint_records():
    records = [{"index": 0, "measurement": {"BLANK": "she"}},
               {"index": 0, "measurement": {"BLANK": "None"}},
               {"index": 0, "measurement": {"BLANK1": "she", "BLANK2": "he"}},
               {"index": 0, "measurement": {"BLANK1": "None", "BLANK2": "None"}},
               {"index": 1, "measurement": {"BLANK": "he"}}]
    assert get_index(0, records) == [records[0], records[2]]
