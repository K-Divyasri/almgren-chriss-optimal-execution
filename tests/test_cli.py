import json

import pytest

from acexec.cli import build_parser, main


def _run(capsys, argv):
    main(argv)
    return capsys.readouterr().out


def test_trajectory_command_prints_holdings(capsys):
    out = _run(capsys, ["--intervals", "5", "trajectory", "--lam", "1e-6"])
    assert "expected cost" in out
    assert "t= 0" in out
    assert "t= 5" in out


def test_frontier_command_prints_table(capsys):
    out = _run(capsys, ["--intervals", "5", "frontier"])
    assert "lam" in out
    assert "expected_cost_bps" in out


def test_frontier_command_json_is_valid(capsys):
    out = _run(capsys, ["--intervals", "5", "frontier", "--json"])
    rows = json.loads(out)
    assert len(rows) > 0
    assert "cost_std_bps" in rows[0]


def test_verify_command_passes_for_default_params(capsys):
    out = _run(capsys, ["--intervals", "6", "verify"])
    assert "MISMATCH" not in out


def test_parser_requires_a_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
