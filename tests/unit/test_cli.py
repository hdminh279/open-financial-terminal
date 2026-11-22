"""Unit tests for the unified Command-Line Interface."""

from unittest.mock import patch

import pytest

from src.cli import build_parser, main


def test_cli_parser_ingest_all():
    """Verify parser parses 'ingest --all' with default options."""
    parser = build_parser()
    args = parser.parse_args(["ingest", "--all"])
    assert args.command == "ingest"
    assert args.all is True
    assert args.facts is False
    assert args.filings is False
    assert args.config == "configs/companies.yaml"
    assert args.limit == 4
    assert args.force is False


def test_cli_parser_ingest_facts_custom_config():
    """Verify parser parses 'ingest --facts' with custom parameters."""
    parser = build_parser()
    args = parser.parse_args(["ingest", "--facts", "--config", "custom.yaml", "--force"])
    assert args.command == "ingest"
    assert args.facts is True
    assert args.all is False
    assert args.config == "custom.yaml"
    assert args.force is True


def test_cli_parser_ingest_vn():
    """Verify parser parses 'ingest --vn' with optional ticker and custom config."""
    parser = build_parser()
    args = parser.parse_args(["ingest", "--vn", "--ticker", "FPT", "--force"])
    assert args.command == "ingest"
    assert args.vn is True
    assert args.ticker == "FPT"
    assert args.force is True


def test_cli_parser_verify():
    """Verify parser parses 'verify' command."""
    parser = build_parser()
    args = parser.parse_args(["verify", "--bucket", "sec-bronze", "--prefix", "raw/filings"])
    assert args.command == "verify"
    assert args.bucket == "sec-bronze"
    assert args.prefix == "raw/filings"


def test_cli_parser_mutually_exclusive_ingest():
    """Verify parser rejects specifying both --facts and --filings."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "--facts", "--filings"])


@patch("src.cli.run_ingest")
def test_cli_main_dispatches_ingest(mock_run_ingest):
    """Verify main dispatches to run_ingest."""
    mock_run_ingest.return_value = 0
    exit_code = main(["ingest", "--all"])
    assert exit_code == 0
    mock_run_ingest.assert_called_once()


@patch("src.cli.run_verify")
def test_cli_main_dispatches_verify(mock_run_verify):
    """Verify main dispatches to run_verify."""
    mock_run_verify.return_value = 0
    exit_code = main(["verify"])
    assert exit_code == 0
    mock_run_verify.assert_called_once()


def test_cli_parser_index():
    """Verify parser parses 'index' subcommand options."""
    parser = build_parser()
    args = parser.parse_args(["index", "--sync", "--verify", "--batch-size", "128"])
    assert args.command == "index"
    assert args.sync is True
    assert args.verify is True
    assert args.batch_size == 128


@patch("src.cli.run_index")
def test_cli_main_dispatches_index(mock_run_index):
    """Verify main dispatches to run_index."""
    mock_run_index.return_value = 0
    exit_code = main(["index", "--sync"])
    assert exit_code == 0
    mock_run_index.assert_called_once()


def test_cli_parser_process():
    """Verify parser parses 'process' subcommand options."""
    parser = build_parser()
    args = parser.parse_args(["process", "--all", "--ticker", "AAPL", "--force"])
    assert args.command == "process"
    assert args.all is True
    assert args.ticker == "AAPL"
    assert args.force is True


@patch("src.cli.run_process")
def test_cli_main_dispatches_process(mock_run_process):
    """Verify main dispatches to run_process."""
    mock_run_process.return_value = 0
    exit_code = main(["process", "--facts"])
    assert exit_code == 0
    mock_run_process.assert_called_once()


def test_cli_parser_process_vn():
    """Verify parser parses 'process --vn' option."""
    parser = build_parser()
    args = parser.parse_args(["process", "--vn", "--ticker", "FPT"])
    assert args.command == "process"
    assert args.vn is True
    assert args.ticker == "FPT"
