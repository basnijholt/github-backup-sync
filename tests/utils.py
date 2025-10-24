"""Shared helpers for testing."""

from __future__ import annotations

import github_backup_sync as gbs


def make_repo(
    *,
    name: str = "repo",
    owner: str = "octocat",
    fork: bool = False,
    archived: bool = False,
    default_branch: str | None = "main",
    ssh_url: str | None = "git@github.com:octocat/repo.git",
    clone_url: str | None = "https://github.com/octocat/repo.git",
) -> gbs.RepoInfo:
    """Return a RepoInfo instance populated with sensible defaults."""
    return gbs.RepoInfo(
        full_name=f"{owner}/{name}",
        name=name,
        owner=owner,
        fork=fork,
        archived=archived,
        ssh_url=ssh_url or "",
        clone_url=clone_url or "",
        default_branch=default_branch,
    )
