"""Tests for internal helper utilities to push coverage over 80%."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from http import HTTPStatus
from pathlib import Path

import pytest

import github_backup_sync as gbs
from tests.utils import make_repo


class DummyProcess:
    """Lightweight stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_repo_info_from_payload() -> None:
    payload = {
        "name": "repo",
        "owner": {"login": "octocat"},
        "full_name": "octocat/repo",
        "fork": True,
        "archived": True,
        "ssh_url": "git@github.com:octocat/repo.git",
        "clone_url": "https://github.com/octocat/repo.git",
        "default_branch": "main",
    }
    info = gbs.RepoInfo.from_payload(payload)
    assert info.full_name == "octocat/repo"
    assert info.fork is True


def test_build_remote_url_https_token_injection() -> None:
    repo = make_repo()
    url = gbs._build_remote_url(repo, use_https=True, token="abc123")
    assert "abc123@" in url


def test_build_remote_url_missing_raises() -> None:
    repo = make_repo(ssh_url=None)
    with pytest.raises(RuntimeError):
        gbs._build_remote_url(repo, use_https=False, token=None)


def test_build_remote_url_https_without_token() -> None:
    repo = make_repo()
    url = gbs._build_remote_url(repo, use_https=True, token=None)
    assert url == repo.clone_url


def test_require_token_or_exit_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs, "_obtain_token", lambda: "token")
    assert gbs._require_token_or_exit() == "token"


def test_require_token_or_exit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs, "_obtain_token", lambda: None)
    with pytest.raises(gbs.typer.Exit):
        gbs._require_token_or_exit()


def test_obtain_token_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs.shutil, "which", lambda _: None)
    assert gbs._obtain_token() is None


def test_obtain_token_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs.shutil, "which", lambda _: "/usr/bin/gh")

    def boom(*_, **__):
        raise subprocess.CalledProcessError(1, "gh")

    monkeypatch.setattr(gbs.subprocess, "run", boom)
    assert gbs._obtain_token() is None


def test_obtain_token_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        gbs.subprocess,
        "run",
        lambda *_, **__: DummyProcess(stdout=""),
    )
    assert gbs._obtain_token() is None


def test_obtain_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        gbs.subprocess,
        "run",
        lambda *_, **__: DummyProcess(stdout="secret-token\n"),
    )
    logs: list[str] = []
    monkeypatch.setattr(gbs.console, "log", lambda *msg, **__: logs.append(" ".join(str(m) for m in msg)))
    assert gbs._obtain_token() == "secret-token"


def test_run_git_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(gbs.subprocess, "run", fake_run)
    gbs._run_git(["status"], cwd=Path("/tmp"), check=False)
    assert recorded["cmd"][0] == "git"


def test_sync_repository_clones_bare_with_lfs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo()
    job = gbs.MirrorJob(repo=repo, destination=tmp_path / "source" / "octocat" / "repo.git", remote=repo.ssh_url)
    commands: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_run_git(args, *, cwd=None, check=True):
        commands.append((tuple(args), Path(cwd) if cwd else None))
        stdout = "cloned" if args[0] == "clone" else ""
        return DummyProcess(stdout=stdout)

    monkeypatch.setattr(gbs, "_run_git", fake_run_git)
    monkeypatch.setattr(gbs.console, "log", lambda *_, **__: None)

    action = gbs._sync_repository(job, gbs.MirrorOptions(bare=True, lfs=True, sleep_seconds=0.0))

    assert action == "cloned"
    assert commands[0][0][:2] == ("clone", "--mirror")
    assert commands[-1][0][0] == "lfs"


def test_sync_repository_updates_working_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo(default_branch=None)
    destination = tmp_path / "source" / "octocat" / "repo"
    destination.mkdir(parents=True)
    job = gbs.MirrorJob(repo=repo, destination=destination, remote=repo.clone_url)
    commands: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_run_git(args, *, cwd=None, check=True):
        commands.append((tuple(args), Path(cwd) if cwd else None))
        if args[0] == "symbolic-ref":
            return DummyProcess(stdout="refs/remotes/origin/main\n")
        if args[0] == "checkout" and len(args) == 2:
            return DummyProcess(stdout="", returncode=0)
        if args[0] == "fetch":
            return DummyProcess(stdout="fetched")
        return DummyProcess()

    monkeypatch.setattr(gbs, "_run_git", fake_run_git)
    monkeypatch.setattr(gbs.console, "log", lambda *_, **__: None)

    action = gbs._sync_repository(job, gbs.MirrorOptions(bare=False, lfs=False, sleep_seconds=0.0))

    assert action == "updated"
    executed = [cmd for cmd, _ in commands]
    assert ("fetch", "--all", "--tags", "--prune") in executed
    assert ("checkout", "main") in executed


def test_sync_repository_updates_bare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo()
    destination = tmp_path / "source" / "octocat" / "repo.git"
    destination.mkdir(parents=True)
    job = gbs.MirrorJob(repo=repo, destination=destination, remote=repo.ssh_url)
    commands: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_run_git(args, *, cwd=None, check=True):
        commands.append((tuple(args), Path(cwd) if cwd else None))
        stdout = "updated" if args[:3] == ("remote", "update", "--prune") else ""
        return DummyProcess(stdout=stdout)

    monkeypatch.setattr(gbs, "_run_git", fake_run_git)
    monkeypatch.setattr(gbs.console, "log", lambda *_, **__: None)

    action = gbs._sync_repository(job, gbs.MirrorOptions(bare=True, lfs=False, sleep_seconds=0.0))
    assert action == "updated"
    assert ("remote", "update", "--prune") in [cmd for cmd, _ in commands]


def test_sync_repository_checkout_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo(default_branch=None)
    destination = tmp_path / "source" / "octocat" / "repo"
    destination.mkdir(parents=True)
    job = gbs.MirrorJob(repo=repo, destination=destination, remote=repo.clone_url)
    commands: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_run_git(args, *, cwd=None, check=True):
        commands.append((tuple(args), Path(cwd) if cwd else None))
        if args[0] == "symbolic-ref":
            return DummyProcess(stdout="refs/remotes/origin/main\n")
        if args[0] == "checkout" and len(args) == 2:
            return DummyProcess(returncode=1)
        return DummyProcess()

    monkeypatch.setattr(gbs, "_run_git", fake_run_git)
    monkeypatch.setattr(gbs.console, "log", lambda *_, **__: None)

    gbs._sync_repository(job, gbs.MirrorOptions(bare=False, lfs=False, sleep_seconds=0.0))
    assert ("checkout", "-B", "main", "origin/main") in [cmd for cmd, _ in commands]


def test_sync_repository_lfs_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo()
    job = gbs.MirrorJob(repo=repo, destination=tmp_path / "source" / "octocat" / "repo.git", remote=repo.ssh_url)

    def fake_run_git(args, *, cwd=None, check=True):
        if args[0] == "clone":
            return DummyProcess()
        if args[0] == "lfs":
            raise FileNotFoundError
        return DummyProcess()

    monkeypatch.setattr(gbs, "_run_git", fake_run_git)
    monkeypatch.setattr(gbs.console, "log", lambda *_, **__: None)

    action = gbs._sync_repository(job, gbs.MirrorOptions(bare=True, lfs=True, sleep_seconds=0.0))
    assert action == "cloned"


def test_mirror_worker_handles_called_process_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo()
    job = gbs.MirrorJob(repo=repo, destination=tmp_path / "repo.git", remote=repo.ssh_url)

    def boom(*_, **__):
        raise subprocess.CalledProcessError(1, "git", output="bad news")

    monkeypatch.setattr(gbs, "_sync_repository", boom)
    result = gbs._mirror_worker(job, gbs.MirrorOptions(bare=True, lfs=False, sleep_seconds=0.0))
    assert result[1] == "error"
    assert "bad news" in result[3]


def test_mirror_worker_respects_sleep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo()
    job = gbs.MirrorJob(repo=repo, destination=tmp_path / "repo.git", remote=repo.ssh_url)
    monkeypatch.setattr(gbs, "_sync_repository", lambda *_: "updated")
    slept: list[float] = []
    monkeypatch.setattr(gbs.time, "sleep", lambda seconds: slept.append(seconds))

    result = gbs._mirror_worker(job, gbs.MirrorOptions(bare=True, lfs=False, sleep_seconds=0.25))
    assert result[1] == "updated"
    assert slept == [0.25]


def test_sync_repository_handles_default_branch_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = make_repo(default_branch=None)
    destination = tmp_path / "source" / "octocat" / "repo"
    destination.mkdir(parents=True)
    job = gbs.MirrorJob(repo=repo, destination=destination, remote=repo.clone_url)

    def fake_run_git(args, *, cwd=None, check=True):
        if args[0] == "symbolic-ref":
            return DummyProcess(stdout="", returncode=1)
        return DummyProcess()

    monkeypatch.setattr(gbs, "_run_git", fake_run_git)
    monkeypatch.setattr(gbs.console, "log", lambda *_, **__: None)

    action = gbs._sync_repository(job, gbs.MirrorOptions(bare=False, lfs=False, sleep_seconds=0.0))
    assert action == "updated"


def test_prune_or_preview_behaviour(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    root = tmp_path
    keep = root / "source" / "keep" / "repo.git"
    stray = root / "forks" / "stray" / "old.git"
    keep.mkdir(parents=True)
    stray.mkdir(parents=True)
    expected = {keep.resolve()}

    gbs._prune_or_preview(root, expected, prune=True)
    assert not stray.exists()

    gbs._prune_or_preview(root, expected, prune=True)
    empty_output = capfd.readouterr().out
    assert "No stale mirrors to prune" in empty_output

    stray2 = root / "forks" / "stray" / "new.git"
    stray2.mkdir(parents=True)
    gbs._prune_or_preview(root, expected, prune=False)
    output = capfd.readouterr().out
    assert "Dry-run prune preview" in output
    assert "new.g" in output

    shutil.rmtree(stray2)
    gbs._prune_or_preview(root, expected, prune=False)
    final_output = capfd.readouterr().out
    assert "No stray local mirrors detected" in final_output


def test_prune_or_preview_skips_non_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    bogus = tmp_path / "source" / "org" / "repo.git"
    bogus.parent.mkdir(parents=True)
    bogus.touch()

    monkeypatch.setattr(gbs, "_collect_existing", lambda _root: {bogus.resolve()})
    gbs._prune_or_preview(tmp_path, set(), prune=True)
    output = capfd.readouterr().out
    assert "Skipping unexpected path" in output


def test_load_repositories_limit_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_a = make_repo(name="a")
    repo_b = make_repo(name="b")

    async def fake_fetch(**_):
        return [repo_a, repo_b]

    async def fake_error(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(gbs, "_fetch_repositories", fake_fetch)
    limited = gbs._load_repositories(token="t", include_archived=True, skip_forks=False, limit=1)
    assert [repo.full_name for repo in limited] == ["octocat/a"]

    monkeypatch.setattr(gbs, "_fetch_repositories", fake_fetch)
    all_repos = gbs._load_repositories(token="t", include_archived=True, skip_forks=False, limit=None)
    assert len(all_repos) == 2

    monkeypatch.setattr(gbs, "_fetch_repositories", fake_error)
    with pytest.raises(gbs.typer.Exit):
        gbs._load_repositories(token="t", include_archived=True, skip_forks=False, limit=None)


def test_fetch_repositories_filters_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        {"name": "a", "owner": {"login": "octo"}, "full_name": "octo/a", "fork": True, "archived": False},
        {"name": "b", "owner": {"login": "octo"}, "full_name": "octo/b", "fork": False, "archived": True},
        {
            "name": "c",
            "owner": {"login": "octo"},
            "full_name": "octo/c",
            "fork": False,
            "archived": False,
            "ssh_url": "git@github.com:octo/c.git",
            "clone_url": "https://github.com/octo/c.git",
            "default_branch": "main",
        },
    ]

    class DummyGitHubAPI:
        def __init__(self, *_args, **_kwargs):
            pass

        async def getiter(self, *_args, **_kwargs):
            for payload in payloads:
                yield payload

    class DummyClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(gbs, "GitHubAPI", DummyGitHubAPI)
    monkeypatch.setattr(gbs.httpx, "AsyncClient", DummyClient)

    repos = asyncio.run(
        gbs._fetch_repositories(
            token="token",
            include_archived=False,
            skip_forks=True,
            limit=None,
        ),
    )
    assert [repo.full_name for repo in repos] == ["octo/c"]

    class ErrorGitHubAPI(DummyGitHubAPI):
        async def getiter(self, *_args, **_kwargs):
            raise gbs.HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR)
            yield  # pragma: no cover

    monkeypatch.setattr(gbs, "GitHubAPI", ErrorGitHubAPI)
    with pytest.raises(RuntimeError):
        asyncio.run(
            gbs._fetch_repositories(
                token="token",
                include_archived=True,
                skip_forks=False,
                limit=None,
            ),
        )


def test_determine_worker_count_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbs.os, "cpu_count", lambda: 24)
    assert gbs._determine_worker_count(None) == 8
    assert gbs._determine_worker_count(3) == 3
