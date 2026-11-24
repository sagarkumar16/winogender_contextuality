"""Find where a collection stopped, and clear the way to continue it without duplicates.

The collectors append one record at a time, so a run that stops early (a time limit, a CUDA error)
leaves the row it was in incomplete, and every row before it complete. This finds the first
incomplete row, removes its records and those of any later row, and prints its number:
collect.sh then continues from there.

    python -m revisions.resume_point OUTPUT.ndjson --per-row 100                  # first incomplete row
    python -m revisions.resume_point OUTPUT.ndjson --per-row 100 --search-from 70 # ... at or after row 70
    python -m revisions.resume_point OUTPUT.ndjson --per-row 100 --from-row 78    # restart at row 78

Prints the row to start from; for a finished file that is one past its last row. A file holding
more than --per-row records for some row was not written by this collection (or already has
duplicates), so it is left alone and the exit status is 1.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def resume_point(path: Path, per_row: int, from_row: int | None = None, search_from: int | None = None) -> int:
    """
    :param path: output file of one collection (records with an integer "index", one per line)
    :param per_row: records a complete row has (e.g. 2 option orders x n_runs for refnull)
    :param from_row: restart at this row instead of the first incomplete one
    :param search_from: look for the first incomplete row from here (default: the file's first row);
                        rows before it are kept as they are
    :return: the row to continue from; records of that row and later ones have been removed
    """
    if not path.exists():
        return next(r for r in (from_row, search_from, 0) if r is not None)

    records = []   # (index, line)
    unreadable = 0
    with open(path) as fh:
        for line in fh:
            try:
                records.append((json.loads(line)["index"], line if line.endswith("\n") else line + "\n"))
            except (json.JSONDecodeError, KeyError, TypeError):
                unreadable += 1   # a line cut off when the writer died
    counts = Counter(index for index, _ in records)

    over = {index: n for index, n in counts.items() if n > per_row}
    if over:
        shown = ", ".join(f"row {i}: {n}" for i, n in sorted(over.items())[:5])
        raise ValueError(f"{path} has more than {per_row} records for {len(over)} rows ({shown}). It is not "
                         f"the output of this collection, or it already holds duplicates; not touching it.")

    if from_row is None:
        # Rows are collected in order, from the first one present
        row = search_from if search_from is not None else min(counts, default=0)
        while counts[row] == per_row:
            row += 1
    else:
        row = from_row

    keep = [line for index, line in records if index < row]
    if len(keep) != len(records) or unreadable:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as fh:
            fh.writelines(keep)
        os.replace(tmp, path)
        print(f"{path.name}: removed {len(records) - len(keep)} records of rows >= {row}"
              + (f" and {unreadable} unreadable lines" if unreadable else ""), file=sys.stderr)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-row", type=int, required=True, help="records in a complete row")
    parser.add_argument("--from-row", type=int, help="restart at this row instead of the first incomplete one")
    parser.add_argument("--search-from", type=int, help="look for the first incomplete row from this row on")
    args = parser.parse_args()
    try:
        print(resume_point(args.output, args.per_row, args.from_row, args.search_from))
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
