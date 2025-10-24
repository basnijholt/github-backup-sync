# GitHub Backup Sync

`github_backup_sync.py` mirrors every GitHub repository your account can access into bare `--mirror` clones, grouping sources and forks separately. The project was inspired after watching [ThePrimeTime's reminder about GitHub bans](https://www.youtube.com/watch?v=7gCCXCSs734), highlighting why local backups matter. Plenty of alternative backup tools exist, but I wanted something simple that relies on the GitHub CLI for authentication so I never have to juggle API tokens directly.

## Requirements

- [`uv`](https://docs.astral.sh/uv/latest/) (recommended; the shebang runs the script through `uv run`)
- GitHub CLI (`gh`) authenticated with `gh auth login`
- `git` and, optionally, `git-lfs` if you use the `--lfs` flag

The script also works inside a manual virtual environment if you install the same dependencies listed in the script header, but `uv` provides the fastest startup and dependency management.

## Usage

```bash
./github_backup_sync.py --root /path/to/mirrors [--https] [--prune] [--skip-forks] [--workers 4]
```

- `--root` points at the backup directory.
- `--https` switches from SSH to HTTPS remotes (still using the `gh` token).
- `--prune` removes local mirrors that no longer exist upstream.
- `--skip-forks` mirrors only non-fork repositories.
- `--workers` controls how many repositories sync in parallel (default: CPU-based).
- `--limit` is handy for smoke-testing with only a few repos.

The script automatically fetches a GitHub token from `gh auth token`, so you only need to keep the CLI logged in.

## Layout

```
<root>/source/<owner>/<repo>.git
<root>/forks/<owner>/<repo>.git
```

All repositories are bare mirrors, suitable for backup purposes.

## Automated Backups (cron)

You can schedule a daily sync via `cron` after installing the script somewhere on your `$PATH`:

```cron
0 3 * * * /usr/bin/env uv run /opt/github-backup-sync/github_backup_sync.py --root /srv/github-backups --https --prune --workers 4 >> /var/log/github-backup-sync.log 2>&1
```

This example runs every day at 03:00, performs a prune, and logs output. Adjust the path, flags, and worker count to suit your environment.
