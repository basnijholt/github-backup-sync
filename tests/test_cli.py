"""Pytest suite exercising the public Typer CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import github_backup_sync as gbs
from tests.utils import make_repo


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_bare_mode_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: CliRunner) -> None:
    repo = make_repo()
    monkeypatch.setattr(gbs, "_require_token_or_exit", lambda: "token")
    monkeypatch.setattr(gbs, "_load_repositories", lambda **_: [repo])

    captured: list[tuple[gbs.MirrorJob, gbs.MirrorOptions]] = []

    def fake_sync(job: gbs.MirrorJob, options: gbs.MirrorOptions) -> str:
        captured.append((job, options))
        return "cloned"

    monkeypatch.setattr(gbs, "_sync_repository", fake_sync)

    result = runner.invoke(gbs.app, ["--root", str(tmp_path / "mirrors"), "--workers", "2"])

    assert result.exit_code == 0
    assert captured
    job, options = captured[0]
    assert options.bare is True
    assert job.destination.suffix == ".git"


def test_cli_working_tree_https_prune(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: CliRunner) -> None:
    repo = make_repo(name="site", owner="org", fork=True, default_branch="develop")
    monkeypatch.setattr(gbs, "_require_token_or_exit", lambda: "secret-token")
    monkeypatch.setattr(gbs, "_load_repositories", lambda **_: [repo])

    stray = tmp_path / "source" / "stray" / "old.git"
    stray.mkdir(parents=True)

    def fake_sync(job: gbs.MirrorJob, options: gbs.MirrorOptions) -> str:
        assert options.bare is False
        assert job.destination.suffix == ""
        raise RuntimeError("transient error")

    monkeypatch.setattr(gbs, "_sync_repository", fake_sync)

    result = runner.invoke(
        gbs.app,
        [
            "--root",
            str(tmp_path),
            "--https",
            "--working-tree",
            "--prune",
            "--lfs",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert not stray.exists()
    assert "Errors encountered" in result.stdout


def test_cli_no_repositories(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: CliRunner) -> None:
    monkeypatch.setattr(gbs, "_require_token_or_exit", lambda: "token")
    monkeypatch.setattr(gbs, "_load_repositories", lambda **_: [])

    result = runner.invoke(gbs.app, ["--root", str(tmp_path / "backups")])

    assert result.exit_code == 0
    assert "No repositories matched" in result.stdout


def test_cli_skips_invalid_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: CliRunner) -> None:
    bad_repo = make_repo(ssh_url=None, clone_url=None)
    monkeypatch.setattr(gbs, "_require_token_or_exit", lambda: "token")
    monkeypatch.setattr(gbs, "_load_repositories", lambda **_: [bad_repo])

    result = runner.invoke(gbs.app, ["--root", str(tmp_path / "root")])

    assert result.exit_code == 0
    assert "No repositories left to process" in result.stdout


def test_cli_requires_root_option(runner: CliRunner) -> None:
    result = runner.invoke(gbs.app, [])
    assert result.exit_code == 1
    assert "Missing required option '--root'" in result.stdout
