<p align="center">
  <img src="https://raw.githubusercontent.com/basnijholt/github-backup-sync/main/.github/logo.svg" alt="GitHub Backup Sync logo" width="160">
</p>

<h1 align="center">GitHub Backup Sync</h1>

<p align="center">
  <em>Mirror every GitHub repository you can access into local git backups.<br>Authenticated through the GitHub CLI, so there are no tokens to juggle.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/github-backup-sync/"><img src="https://img.shields.io/pypi/v/github-backup-sync?label=PyPI&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://pypi.org/project/github-backup-sync/"><img src="https://img.shields.io/pypi/pyversions/github-backup-sync?logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://pypistats.org/packages/github-backup-sync"><img src="https://img.shields.io/pypi/dm/github-backup-sync?label=downloads" alt="Monthly downloads"></a>
  <a href="https://github.com/basnijholt/github-backup-sync/actions/workflows/pytest-uv.yml"><img src="https://img.shields.io/github/actions/workflow/status/basnijholt/github-backup-sync/pytest-uv.yml?branch=main&label=tests&logo=github" alt="Tests"></a>
  <a href="https://codecov.io/gh/basnijholt/github-backup-sync"><img src="https://img.shields.io/codecov/c/github/basnijholt/github-backup-sync?logo=codecov&logoColor=white" alt="Coverage"></a>
  <a href="https://github.com/basnijholt/github-backup-sync/blob/main/LICENSE"><img src="https://img.shields.io/github/license/basnijholt/github-backup-sync" alt="License"></a>
</p>

`github-backup-sync` mirrors every GitHub repository your account can access (owned, collaborator, and organization member) into bare `--mirror` clones, grouping sources and forks separately.

The project was inspired after watching [ThePrimeTime's reminder about GitHub bans](https://www.youtube.com/watch?v=7gCCXCSs734), highlighting why local backups matter.
Plenty of alternative backup tools exist, but I wanted something simple that relies on the GitHub CLI for authentication so I never have to juggle API tokens directly.

<details>
<summary><b>Table of Contents</b></summary>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Features](#features)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Options](#options)
  - [CLI Help](#cli-help)
- [Layout](#layout)
- [Working with Mirrors](#working-with-mirrors)
- [Automated Backups (cron)](#automated-backups-cron)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

</details>

## Features

- **Complete mirrors**: bare `git clone --mirror` copies keep every branch, tag, and ref; re-runs update them in place.
- **Sources and forks apart**: forks land in their own directory, or skip them entirely with `--skip-forks`.
- **No token management**: the token comes from `gh auth token`, so staying logged in to the GitHub CLI is all it takes.
- **Fast**: repositories sync in parallel (`--workers`).
- **Safe pruning**: mirrors of repositories that disappeared upstream are listed on every run and only deleted with `--prune`.
- **Flexible**: optional working-tree checkouts, Git LFS objects, and archived-repository filtering.
- **Single file**: one script with a `uv run --script` shebang, so you can also vendor it anywhere.

## Quick Start

```bash
gh auth login                                  # once
uvx github-backup-sync --root ~/github-backups # mirror everything
```

Run the same command again whenever you want to pull in new commits, branches, and repositories.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) (recommended) or any Python 3.12+ environment
- [GitHub CLI](https://cli.github.com/) (`gh`) authenticated with `gh auth login`
- `git`, plus `git-lfs` if you use `--lfs`
- SSH access to GitHub for the default SSH remotes, or pass `--https`

## Installation

| Method                 | Command                                                        |
| ---------------------- | -------------------------------------------------------------- |
| Run without installing | `uvx github-backup-sync --help`                                |
| Install as a tool      | `uv tool install github-backup-sync`                           |
| Install with pip       | `pip install github-backup-sync`                               |
| Vendor the script      | Copy `github_backup_sync.py` and run `./github_backup_sync.py` |

`uvx` fetches the latest release on each run, while `uv tool install` puts a `github-backup-sync` executable on your `PATH`.
The vendored script installs its own dependencies through the `uv run --script` shebang.

## Usage

```bash
# If installed from PyPI
github-backup-sync --root /path/to/mirrors [--https] [--prune] [--skip-forks] [--workers 4]

# From a local checkout
./github_backup_sync.py --root /path/to/mirrors [--https] [--prune] [--skip-forks] [--workers 4]
```

### Options

| Option                                    | Description                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `-r, --root PATH`                         | Backup directory (required, created if missing)                                     |
| `--https`                                 | Use HTTPS remotes instead of SSH; the `gh` token authenticates private repositories |
| `--prune`                                 | Delete local mirrors that no longer exist upstream; without it they are only listed |
| `--working-tree / --bare`                 | Clone with files checked out instead of bare mirrors (default: `--bare`)            |
| `--include-archived / --exclude-archived` | Whether to back up archived repositories (default: include)                         |
| `--skip-forks`                            | Mirror only non-fork repositories                                                   |
| `--lfs`                                   | Fetch Git LFS objects after mirroring                                               |
| `-w, --workers N`                         | Parallel sync operations (default: CPU count, capped at 8)                          |
| `--sleep SECONDS`                         | Delay between repositories                                                          |
| `--limit N`                               | Process at most `N` repositories, handy for smoke tests                             |

> [!WARNING]
> With `--https`, the token is embedded in each remote URL, so it is stored in plain text in every mirror's `config` file.
> Keep the backup directory private, or stick with the default SSH remotes.

> [!NOTE]
> In `--working-tree` mode every run checks out the default branch and hard-resets it to `origin`, discarding local edits.
> Treat those checkouts as read-only copies.

### CLI Help

<details>
<summary><code>github_backup_sync.py --help</code> output</summary>

<!-- CODE:BASH:START -->
<!-- echo '```text' -->
<!-- export NO_COLOR=1 -->
<!-- export TERM=dumb -->
<!-- export TERMINAL_WIDTH=110 -->
<!-- ./github_backup_sync.py --help -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```text

 Usage: github_backup_sync.py [OPTIONS]

 Mirror every GitHub repository you can access into local git mirrors.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --root            -r                      <path>           Directory that    │
│                                                            will hold the     │
│                                                            mirror            │
│ --https                                                    Use HTTPS remotes │
│                                                            (token optional   │
│                                                            for public repos) │
│ --include-archi…      --exclude-archi…                     Include archived  │
│                                                            repositories      │
│                                                            [default:         │
│                                                            include-archived] │
│ --prune                                                    Remove local      │
│                                                            mirrors that no   │
│                                                            longer exist      │
│                                                            upstream          │
│ --lfs                                                      Fetch Git LFS     │
│                                                            objects after     │
│                                                            mirroring         │
│ --working-tree        --bare                               Clone             │
│                                                            repositories as   │
│                                                            working trees     │
│                                                            instead of bare   │
│                                                            mirrors           │
│                                                            [default: bare]   │
│ --sleep                                   <float range>    Delay between     │
│                                           [x>=0.0]         repositories in   │
│                                                            seconds           │
│                                                            [default: 0.0]    │
│ --limit                                   <int range>      Process at most   │
│                                           [x>=1]           this many         │
│                                                            repositories      │
│ --skip-forks                                               Ignore forked     │
│                                                            repositories      │
│                                                            while mirroring   │
│ --workers         -w                      <int range>      Max concurrent    │
│                                           [x>=1]           mirror operations │
│                                                            (default: based   │
│                                                            on CPU count)     │
│ --help            -h                                       Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯

```

<!-- OUTPUT:END -->

</details>

## Layout

```
<root>/source/<owner>/<repo>.git
<root>/forks/<owner>/<repo>.git
```

This is the default layout when backing up as bare mirrors.
When `--working-tree` is used, the `.git` suffix is omitted and each directory contains a checked-out working tree instead.

## Working with Mirrors

Repositories under `source/` and `forks/` are bare `--mirror` clones, so they do not contain a checked-out working tree unless you opted into `--working-tree`.
Clone from a mirror path when you want editable files:

```bash
git clone /srv/github-backups/source/basnijholt/dotfiles.git ~/dotfiles
```

You can also inspect a mirror directly without cloning by pointing Git commands at it:

```bash
git --git-dir=/srv/github-backups/source/basnijholt/dotfiles.git log --oneline
```

## Automated Backups (cron)

Schedule a daily sync with `cron`:

```cron
0 3 * * * /usr/bin/env uvx github-backup-sync --root /srv/github-backups --https --prune --workers 4 >> /var/log/github-backup-sync.log 2>&1
```

This example runs every day at 03:00, prunes mirrors of deleted repositories, and logs the output.
The cron user needs `uv` on its `PATH` and a logged-in `gh`.
Adjust the path, flags, and worker count to suit your environment.

## License

[MIT](https://github.com/basnijholt/github-backup-sync/blob/main/LICENSE)
