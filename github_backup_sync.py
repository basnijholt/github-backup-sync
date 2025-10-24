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

"""CLI for mirroring accessible GitHub repositories as bare git mirrors."""

import asyncio
import os
import shutil
import subprocess
import time
import urllib.parse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    """Minimal subset of repository metadata required for mirroring."""

    full_name: str
    name: str
    owner: str
    fork: bool
    archived: bool
    ssh_url: str
    clone_url: str
    default_branch: str | None

    @classmethod
    def from_payload(cls, payload: dict) -> "RepoInfo":
        """Create a `RepoInfo` instance from raw GitHub API data."""
        owner = payload.get("owner", {}).get("login", "")
        return cls(
            full_name=payload.get("full_name", f"{owner}/{payload.get('name', '')}"),
            name=payload.get("name", ""),
            owner=owner,
            fork=payload.get("fork", False),
            archived=payload.get("archived", False),
            ssh_url=payload.get("ssh_url", ""),
            clone_url=payload.get("clone_url", ""),
            default_branch=payload.get("default_branch"),
        )


@dataclass(frozen=True)
class MirrorJob:
    """Input describing a single mirroring task."""

    repo: RepoInfo
    destination: Path
    remote: str


@dataclass(frozen=True)
class MirrorOptions:
    """Runtime options that influence mirroring behaviour."""

    bare: bool
    lfs: bool
    sleep_seconds: float


def _ensure_root_directory(ctx: typer.Context, root_option: Path | None) -> Path:
    """Validate the ``--root`` option and ensure the directory exists."""
    if root_option is None:
        typer.echo(ctx.get_help())
        console.print("[red]Missing required option '--root'.[/]")
        raise typer.Exit(1)
    root_path = root_option.expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def _determine_worker_count(workers_option: int | None) -> int:
    """Return the number of workers to use for mirroring."""
    if workers_option is not None:
        return workers_option
    cpu_count = os.cpu_count() or 4
    return min(8, max(1, cpu_count))


def _require_token_or_exit() -> str:
    """Retrieve a GitHub token and exit if it cannot be obtained."""
    token = _obtain_token()
    if not token:
        console.print("[red]Unable to retrieve GitHub token via gh CLI. Exiting.")
        raise typer.Exit(1)
    return token


def _prepare_jobs(
    root: Path,
    repos: list[RepoInfo],
    *,
    bare: bool,
    use_https: bool,
    token: str,
) -> tuple[list[MirrorJob], set[Path]]:
    """Build ``MirrorJob`` objects plus the expected on-disk paths."""
    jobs: list[MirrorJob] = []
    expected_paths: set[Path] = set()
    for repo in repos:
        destination = _desired_path(root, repo, bare=bare)
        expected_paths.add(destination.resolve())
        try:
            remote = _build_remote_url(
                repo,
                use_https=use_https,
                token=token if use_https else None,
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}")
            continue
        jobs.append(MirrorJob(repo, destination, remote))
    return jobs, expected_paths


def _run_mirror_jobs(
    jobs: list[MirrorJob],
    options: MirrorOptions,
    workers: int,
) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str]]]:
    """Run mirror jobs concurrently and collect outcomes."""
    outcomes: list[tuple[str, str, Path]] = []
    errors: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Queued", total=len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_mirror_worker, job, options): job.repo for job in jobs}
            for future in as_completed(future_map):
                name, action, path, error_message = future.result()
                progress.update(task, advance=1, description=name)
                outcomes.append((name, action, path))
                if error_message:
                    errors.append((name, error_message))

    return outcomes, errors


def _render_summary(outcomes: list[tuple[str, str, Path]]) -> None:
    """Render a table summarising mirror actions."""
    table = Table("Repository", "Action", "Path", title="Mirror summary")
    for name, action, path in outcomes:
        table.add_row(name, action, str(path))
    console.print(table)


def _render_errors(errors: list[tuple[str, str]]) -> None:
    """Print any errors collected during mirroring."""
    if not errors:
        return
    console.print("[red]Errors encountered:[/]")
    for name, message in errors:
        console.print(f"  [bold]{name}[/]: {message}")


def _load_repositories(
    *,
    token: str,
    include_archived: bool,
    skip_forks: bool,
    limit: int | None,
) -> list[RepoInfo]:
    """Fetch repositories once, handling CLI-friendly error reporting."""
    try:
        repositories = asyncio.run(
            _fetch_repositories(
                token=token,
                include_archived=include_archived,
                skip_forks=skip_forks,
                limit=limit,
            ),
        )
    except RuntimeError as error:
        console.print(f"[red]{error}")
        raise typer.Exit(1)
    if limit:
        return repositories[:limit]
    return repositories


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a git command and return the completed process."""
    return subprocess.run(  # noqa: S603  # Running git CLI with explicit arguments
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def _obtain_token() -> str | None:
    """Return a GitHub token via the `gh` CLI or `None` if unavailable."""
    gh_executable = shutil.which("gh")
    if not gh_executable:
        console.log("GitHub CLI not found; cannot retrieve token")
        return None
    try:
        console.log("Attempting to retrieve GitHub token via gh auth token")
        proc = subprocess.run(  # noqa: S603  # Running trusted gh CLI command
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


def _build_remote_url(repo: RepoInfo, use_https: bool, token: str | None) -> str:
    """Build the remote URL used for mirroring a repository."""
    if use_https:
        if not repo.clone_url:
            message = f"{repo.full_name}: missing HTTPS clone URL"
            raise RuntimeError(message)
        if token:
            parsed = urllib.parse.urlparse(repo.clone_url)
            netloc = f"{token}@{parsed.netloc}"
            return urllib.parse.urlunparse(parsed._replace(netloc=netloc))
        return repo.clone_url
    if not repo.ssh_url:
        message = f"{repo.full_name}: missing SSH URL"
        raise RuntimeError(message)
    return repo.ssh_url


def _detect_default_branch(repo: RepoInfo, path: Path) -> str | None:
    """Return the default branch for the repository if known."""
    if repo.default_branch:
        return repo.default_branch
    result = _run_git(
        ["symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=path,
        check=False,
    )
    if result.returncode != 0:
        return None
    stdout = cast(str, result.stdout)
    ref = stdout.strip()
    if ref.startswith("refs/remotes/origin/"):
        return ref.rsplit("/", 1)[-1]
    return None


def _clone_initial_repo(job: MirrorJob, bare: bool) -> None:
    """Clone a repository either as a mirror or a working tree."""
    repo = job.repo
    path = job.destination
    remote = job.remote
    result = _run_git(["clone", "--mirror", remote, str(path)]) if bare else _run_git(["clone", remote, str(path)])
    if result.stdout.strip():
        console.log(result.stdout.strip())
    mode = "mirror" if bare else "working tree"
    console.log(f"{repo.full_name}: {mode} cloned")


def _update_existing_repo(job: MirrorJob, bare: bool) -> None:
    """Update an existing clone."""
    repo = job.repo
    path = job.destination
    remote = job.remote
    _run_git(["remote", "set-url", "origin", remote], cwd=path, check=False)
    if bare:
        result = _run_git(["remote", "update", "--prune"], cwd=path)
        if result.stdout.strip():
            console.log(result.stdout.strip())
        console.log(f"{repo.full_name}: mirror updated")
        return

    result = _run_git(["fetch", "--all", "--tags", "--prune"], cwd=path)
    if result.stdout.strip():
        console.log(result.stdout.strip())
    default_branch = _detect_default_branch(repo, path)
    if default_branch:
        checkout = _run_git(["checkout", default_branch], cwd=path, check=False)
        if checkout.returncode != 0:
            _run_git(
                ["checkout", "-B", default_branch, f"origin/{default_branch}"],
                cwd=path,
            )
        _run_git(["reset", "--hard", f"origin/{default_branch}"], cwd=path)
        console.log(f"{repo.full_name}: working tree updated")
        return

    console.log(
        f"{repo.full_name}: unable to determine default branch; skipped worktree reset",
    )


def _sync_repository(job: MirrorJob, options: MirrorOptions) -> str:
    """Clone or update the local mirror for a single repository."""
    path = job.destination
    path.parent.mkdir(parents=True, exist_ok=True)
    action = "updated" if path.exists() else "cloned"
    if path.exists():
        _update_existing_repo(job, options.bare)
    else:
        _clone_initial_repo(job, options.bare)
    if options.lfs:
        try:
            _run_git(["lfs", "fetch", "--all"], cwd=path, check=False)
        except FileNotFoundError:
            console.log("git-lfs not installed; skipping LFS fetch")
    return action


def _mirror_worker(
    job: MirrorJob,
    options: MirrorOptions,
) -> tuple[str, str, Path, str | None]:
    """Mirror a repository and return the action, propagating any errors."""
    try:
        action = _sync_repository(job, options)
    except subprocess.CalledProcessError as exc:
        message = exc.stdout.strip() if exc.stdout else str(exc)
        return job.repo.full_name, "error", job.destination, message
    except Exception as exc:  # noqa: BLE001
        return job.repo.full_name, "error", job.destination, str(exc)
    else:
        if options.sleep_seconds:
            time.sleep(options.sleep_seconds)
        return job.repo.full_name, action, job.destination, None


def _collect_existing(root: Path) -> set[Path]:
    """Return a set of existing mirror directories beneath ``root``."""
    existing: set[Path] = set()
    for category in ("source", "forks"):
        base = root / category
        if not base.exists():
            continue
        for owner_dir in base.iterdir():
            if not owner_dir.is_dir():
                continue
            for repo_dir in owner_dir.iterdir():
                if repo_dir.is_dir():
                    existing.add(repo_dir.resolve())
    return existing


async def _fetch_repositories(
    token: str | None,
    include_archived: bool,
    skip_forks: bool,
    limit: int | None,
) -> list[RepoInfo]:
    """Return repositories accessible to the authenticated user."""
    timeout = httpx.Timeout(10.0, read=30.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"Accept": "application/vnd.github+json"},
    ) as client:
        gh = GitHubAPI(client, "mirror-github-all", oauth_token=token)
        repos: list[RepoInfo] = []
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
                if skip_forks and info.fork:
                    continue
                if not include_archived and info.archived:
                    continue
                repos.append(info)
                if limit and len(repos) >= limit:
                    break
        except HTTPException as exc:
            message = exc.args[0] if exc.args else ""
            error_message = f"GitHub API error: {message}"
            raise RuntimeError(error_message) from exc
    return repos


def _desired_path(root: Path, repo: RepoInfo, bare: bool) -> Path:
    """Compute the target directory for ``repo`` based on mirror mode."""
    category = "forks" if repo.fork else "source"
    suffix = ".git" if bare else ""
    name = f"{repo.name}{suffix}"
    return root / category / repo.owner / name


def _prune_or_preview(root: Path, expected: set[Path], prune: bool) -> None:
    """Either remove or list local mirror directories not in ``expected``."""
    existing = _collect_existing(root)
    to_delete = sorted(existing - expected)
    if prune:
        if not to_delete:
            console.print("[green]No stale mirrors to prune.[/]")
            return
        for path in to_delete:
            if not path.is_dir():
                console.print(f"[yellow]Skipping unexpected path: {path}")
                continue
            console.print(f"[red]Deleting[/] {path}")
            subprocess.run(["rm", "-rf", str(path)], check=False)  # noqa: S603  # Removing stale mirrors from local disk
        console.print(f"[green]Pruned {len(to_delete)} mirror(s).")
    elif to_delete:
        console.print("[cyan]Dry-run prune preview (use --prune to remove):")
        for path in to_delete:
            console.print(f"  {path}")
    else:
        console.print("[green]No stray local mirrors detected.")


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def main(  # noqa: PLR0913  # CLI entrypoint needs many options and branches
    ctx: typer.Context,
    root: Path | None = typer.Option(
        None,
        "--root",
        "-r",
        help="Directory that will hold the mirror",
    ),
    use_https: bool = typer.Option(
        False,
        "--https",
        help="Use HTTPS remotes (token optional for public repos)",
    ),
    include_archived: bool = typer.Option(
        True,
        "--include-archived/--exclude-archived",
        help="Include archived repositories",
    ),
    prune: bool = typer.Option(
        False,
        "--prune",
        help="Remove local mirrors that no longer exist upstream",
    ),
    lfs: bool = typer.Option(
        False,
        "--lfs",
        help="Fetch Git LFS objects after mirroring",
    ),
    working_tree: bool = typer.Option(
        False,
        "--working-tree/--bare",
        help="Clone repositories as working trees instead of bare mirrors",
    ),
    sleep_seconds: float = typer.Option(
        0.0,
        "--sleep",
        min=0.0,
        help="Delay between repositories in seconds",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process at most this many repositories",
    ),
    skip_forks: bool = typer.Option(
        False,
        "--skip-forks",
        help="Ignore forked repositories while mirroring",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        "-w",
        min=1,
        help="Max concurrent mirror operations (default: based on CPU count)",
    ),
) -> None:
    """Coordinate the CLI workflow for mirroring repositories."""
    root_path = _ensure_root_directory(ctx, root)
    bare = not working_tree
    token = _require_token_or_exit()

    repositories = _load_repositories(
        token=token,
        include_archived=include_archived,
        skip_forks=skip_forks,
        limit=limit,
    )

    if not repositories:
        console.print("[yellow]No repositories matched the criteria.")
        raise typer.Exit

    jobs, expected_paths = _prepare_jobs(
        root_path,
        repositories,
        bare=bare,
        use_https=use_https,
        token=token,
    )

    if not jobs:
        console.print("[yellow]No repositories left to process after filtering.")
        raise typer.Exit

    worker_count = _determine_worker_count(workers)
    mode_description = "bare mirrors" if bare else "working trees"
    console.print(
        f"[bold]Mirroring {len(jobs)} repositories into {root_path}[/] as {mode_description} using {worker_count} worker(s)",
    )

    options = MirrorOptions(bare=bare, lfs=lfs, sleep_seconds=sleep_seconds)
    outcomes, errors = _run_mirror_jobs(jobs, options, worker_count)
    _render_summary(outcomes)
    _render_errors(errors)
    _prune_or_preview(root_path, expected_paths, prune=prune)


if __name__ == "__main__":
    app()
