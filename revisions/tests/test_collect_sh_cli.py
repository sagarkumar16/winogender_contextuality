"""The collectors' CLI accepts exactly the arguments revisions/slurm/collect.sh passes."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from typer.testing import CliRunner

from winogender_contextuality.modeling import collect_sequential as cs
from winogender_contextuality.utils import load_ndjson


@pytest.mark.parametrize("command, input_file, extra", [
    ("generate-one-pronoun", "pairs.tsv", []),
    ("generate-one-null-context", "all_sentences.csv", []),
    ("generate-one-null-context", "refnull.csv", ["--prime-column", "prime", "--tag", "refnull"]),
    ("generate-two-pronouns", "pairs.tsv", []),
])
def test_cli_accepts_the_arguments_collect_sh_passes(data_dir, use_fake, command, input_file, extra):
    use_fake()
    out = data_dir / "cli.ndjson"
    result = CliRunner().invoke(cs.app, [
        command, "gpu", "fake/model", "0.5", "--n-runs=1", "--start", "0", "--end", "0",
        "--input-dir", str(data_dir), "--input-file", input_file,
        "--output-file", str(out), "--no-quantized", *extra])
    assert result.exit_code == 0, result.output
    assert len(load_ndjson(out)) > 0
