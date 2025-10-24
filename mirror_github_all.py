#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx>=0.27.0",
#     "gidgethub[httpx]>=5.0.0",
#     "rich>=13.7.0",
#     "typer>=0.12.3"
# ]
# ///

import asyncio
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

import httpx
import typer
from gidgethub import HTTPException
from gidgethub.httpx import GitHubAPI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console(highlight=False)
app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@dataclass(frozen=True)
class RepoInfo:
    full_name: str
    name: str
    owner: str
    fork: bool
    archived: bool
    ssh_url: str
    clone_url: str

    @classmethod
    def from_payload(cls, payload: dict) -> "RepoInfo":
        owner = payload.get("owner", {}).get("login", "")
        return cls(
            full_name=payload.get("full_name", f"{owner}/{payload.get('name', '')}"),
            name=payload.get("name", ""),
            owner=owner,
            fork=payload.get("fork", False),
            archived=payload.get("archived", False),
            ssh_url=payload.get("ssh_url", ""),
            clone_url=payload.get("clone_url", ""),
        )


def run_git(
    args: Sequence[str], cwd: Optional[Path] = None, check: bool = True
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )
    return result


def obtain_token() -> Optional[str]:
    gh_executable = shutil.which("gh")
    if not gh_executable:
        console.log("GitHub CLI not found; cannot retrieve token")
        return None
    try:
        console.log("Attempting to retrieve GitHub token via gh auth token")
        proc = subprocess.run(
            [gh_executable, "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "(no stderr)"
        console.log(f"Failed to retrieve token from gh CLI: {stderr}")
        return None
    token = proc.stdout.strip()
    if not token:
        console.log("gh auth token returned empty output")
        return None
    console.log("Obtained GitHub token via gh CLI")
    return token


def remote_url(repo: RepoInfo, use_https: bool, token: Optional[str]) -> str:
    if use_https:
        if not repo.clone_url:
            raise RuntimeError(f"{repo.full_name}: missing HTTPS clone URL")
        if token:
            parsed = urllib.parse.urlparse(repo.clone_url)
            netloc = f"{token}@{parsed.netloc}"
            return urllib.parse.urlunparse(parsed._replace(netloc=netloc))
        return repo.clone_url
    if not repo.ssh_url:
        raise RuntimeError(f"{repo.full_name}: missing SSH URL")
    return repo.ssh_url


def ensure_mirror(repo: RepoInfo, path: Path, remote: str, lfs: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        run_git(["remote", "set-url", "origin", remote], cwd=path, check=False)
        result = run_git(["remote", "update", "--prune"], cwd=path)
        if result.stdout.strip():
            console.log(result.stdout.strip())
        action = "updated"
    else:
        result = run_git(["clone", "--mirror", remote, str(path)])
        if result.stdout.strip():
            console.log(result.stdout.strip())
        action = "cloned"
    if lfs:
        try:
            run_git(["lfs", "fetch", "--all"], cwd=path, check=False)
        except FileNotFoundError:
            console.log("git-lfs not installed; skipping LFS fetch")
    return action


def collect_existing(root: Path) -> Set[Path]:
    existing: Set[Path] = set()
    for category in ("source", "forks"):
        base = root / category
        if not base.exists():
            continue
        for owner_dir in base.iterdir():
            if not owner_dir.is_dir():
                continue
            for repo_dir in owner_dir.iterdir():
                if repo_dir.is_dir() and repo_dir.name.endswith(".git"):
                    existing.add(repo_dir.resolve())
    return existing


async def fetch_repositories(
    token: Optional[str],
    include_archived: bool,
    limit: Optional[int],
) -> List[RepoInfo]:
    timeout = httpx.Timeout(10.0, read=30.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"Accept": "application/vnd.github+json"},
    ) as client:
        gh = GitHubAPI(client, "mirror-github-all", oauth_token=token)
        repos: List[RepoInfo] = []
        try:
            params = {
                "per_page": 100,
                "affiliation": "owner,collaborator,organization_member",
                "visibility": "all",
                "sort": "full_name",
                "direction": "asc",
            }
            async for payload in gh.getiter("/user/repos", params):
                info = RepoInfo.from_payload(payload)
                if not include_archived and info.archived:
                    continue
                repos.append(info)
                if limit and len(repos) >= limit:
                    break
        except HTTPException as exc:
            message = exc.args[0] if exc.args else ""
            raise RuntimeError(f"GitHub API error: {message}") from exc
    return repos


def desired_path(root: Path, repo: RepoInfo) -> Path:
    category = "forks" if repo.fork else "source"
    return root / category / repo.owner / f"{repo.name}.git"


def prune_or_preview(root: Path, expected: Set[Path], prune: bool) -> None:
    existing = collect_existing(root)
    to_delete = sorted(existing - expected)
    if prune:
        if not to_delete:
            console.print("[green]No stale mirrors to prune.[/]")
            return
        for path in to_delete:
            if not path.is_dir() or not path.name.endswith(".git"):
                console.print(f"[yellow]Skipping unexpected path: {path}")
                continue
            console.print(f"[red]Deleting[/] {path}")
            subprocess.run(["rm", "-rf", str(path)], check=False)
        console.print(f"[green]Pruned {len(to_delete)} mirror(s).")
    else:
        if to_delete:
            console.print("[cyan]Dry-run prune preview (use --prune to remove):")
            for path in to_delete:
                console.print(f"  {path}")
        else:
            console.print("[green]No stray local mirrors detected.")


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def main(
    ctx: typer.Context,
    root: Optional[Path] = typer.Option(
        None, "--root", "-r", help="Directory that will hold the mirror"
    ),
    use_https: bool = typer.Option(
        False, "--https", help="Use HTTPS remotes (token optional for public repos)"
    ),
    include_archived: bool = typer.Option(
        True,
        "--include-archived/--exclude-archived",
        help="Include archived repositories",
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Remove local mirrors that no longer exist upstream"
    ),
    lfs: bool = typer.Option(
        False, "--lfs", help="Fetch Git LFS objects after mirroring"
    ),
    sleep_seconds: float = typer.Option(
        0.0, "--sleep", min=0.0, help="Delay between repositories in seconds"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", min=1, help="Process at most this many repositories"
    ),
) -> None:
    if root is None:
        typer.echo(ctx.get_help())
        console.print("[red]Missing required option '--root'.[/]")
        raise typer.Exit(1)

    token = obtain_token()
    if not token:
        console.print("[red]Unable to retrieve GitHub token via gh CLI. Exiting.")
        raise typer.Exit(1)

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        repos_info = asyncio.run(
            fetch_repositories(
                token=token,
                include_archived=include_archived,
                limit=limit,
            )
        )
    except RuntimeError as error:
        console.print(f"[red]{error}")
        raise typer.Exit(1)

    if not repos_info:
        console.print("[yellow]No repositories matched the criteria.")
        raise typer.Exit()

    if limit:
        repos_info = repos_info[:limit]

    console.print(f"[bold]Mirroring {len(repos_info)} repositories into {root}[/]")

    expected_paths: Set[Path] = set()
    outcomes: List[tuple[str, str, Path]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Preparing", total=len(repos_info))
        for repo in repos_info:
            progress.update(task, description=f"{repo.full_name}")
            destination = desired_path(root, repo)
            expected_paths.add(destination.resolve())
            remote = remote_url(
                repo, use_https=use_https, token=token if use_https else None
            )
            try:
                action = ensure_mirror(repo, destination, remote, lfs=lfs)
                outcomes.append((repo.full_name, action, destination))
            except subprocess.CalledProcessError as exc:
                console.print(f"[red]git failed for {repo.full_name}[/]")
                if exc.stdout:
                    console.print(exc.stdout)
            progress.advance(task)
            if sleep_seconds:
                time.sleep(sleep_seconds)

    table = Table("Repository", "Action", "Path", title="Mirror summary")
    for name, action, path in outcomes:
        table.add_row(name, action, str(path))
    console.print(table)

    prune_or_preview(root, expected_paths, prune=prune)


if __name__ == "__main__":
    app()
