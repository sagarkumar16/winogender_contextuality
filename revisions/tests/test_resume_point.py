"""revisions/resume_point.py: where collect.sh continues a stopped collection, without duplicates."""

import json

import pytest

from revisions.resume_point import resume_point


def write(path, rows, tail=""):
    """rows: {index: number of records}, written in index order like the collectors do."""
    with open(path, "w") as fh:
        for index, n in rows.items():
            for k in range(n):
                fh.write(json.dumps({"index": index, "k": k}) + "\n")
        fh.write(tail)


def indices(path):
    return [json.loads(line)["index"] for line in open(path)]


def test_missing_file_starts_at_zero_or_the_requested_row(tmp_path):
    assert resume_point(tmp_path / "none.ndjson", per_row=4) == 0
    assert resume_point(tmp_path / "none.ndjson", per_row=4, from_row=7) == 7
    assert resume_point(tmp_path / "none.ndjson", per_row=4, search_from=18) == 18


def test_search_from_keeps_earlier_rows_and_finds_the_partial_one(tmp_path):
    # a batch (rows 2-3) that failed in row 3; collect.sh retries from there
    out = tmp_path / "out.ndjson"
    write(out, {0: 4, 1: 4, 2: 4, 3: 1})
    assert resume_point(out, per_row=4, search_from=2) == 3
    assert indices(out) == [0] * 4 + [1] * 4 + [2] * 4
    # a batch that failed before writing anything retries from its own first row
    assert resume_point(out, per_row=4, search_from=3) == 3


def test_drops_the_partial_row_and_continues_there(tmp_path):
    out = tmp_path / "out.ndjson"
    write(out, {0: 4, 1: 4, 2: 3})
    assert resume_point(out, per_row=4) == 2
    assert indices(out) == [0] * 4 + [1] * 4


def test_a_line_cut_off_mid_write_is_dropped(tmp_path):
    out = tmp_path / "out.ndjson"
    write(out, {0: 4, 1: 4}, tail='{"index": 2, "mea')
    assert resume_point(out, per_row=4) == 2
    assert indices(out) == [0] * 4 + [1] * 4
    assert open(out).read().endswith("}\n")


def test_finished_file_is_left_alone(tmp_path):
    out = tmp_path / "out.ndjson"
    write(out, {0: 4, 1: 4, 2: 4})
    before = out.read_bytes()
    assert resume_point(out, per_row=4) == 3
    assert out.read_bytes() == before


def test_from_row_drops_that_row_and_later(tmp_path):
    out = tmp_path / "out.ndjson"
    write(out, {0: 4, 1: 4, 2: 4})
    assert resume_point(out, per_row=4, from_row=1) == 1
    assert indices(out) == [0] * 4


def test_a_file_started_past_row_zero_continues_from_its_own_rows(tmp_path):
    out = tmp_path / "out.ndjson"
    write(out, {50: 4, 51: 2})
    assert resume_point(out, per_row=4) == 51
    assert indices(out) == [50] * 4


def test_refuses_a_file_with_more_records_per_row(tmp_path):
    # e.g. the paper's null file (400 per row) under the name a rerun with 300 per row would use
    out = tmp_path / "out.ndjson"
    write(out, {0: 6, 1: 6})
    before = out.read_bytes()
    with pytest.raises(ValueError, match="not touching it"):
        resume_point(out, per_row=4)
    assert out.read_bytes() == before
