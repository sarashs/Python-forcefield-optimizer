# conversations/

Archived Claude Code session transcripts for this project, kept in
the repo so any machine that clones it can replay or continue them.

Two parallel forms of the same content:

| File | Purpose |
|------|---------|
| `*.md` (this directory) | Human-readable filtered conversation (user prompts + assistant prose, ~500 KB each) — open in any editor. |
| `transcripts/*.jsonl.gz` | Raw Claude Code session log, gzipped (22 MB → ~5 MB each). The format Claude Code's `/resume` reads. |

## How `/resume` finds these

Claude Code's resume picker enumerates `.jsonl` files in
`~/.claude/projects/<encoded-cwd>/` where the *encoded cwd* is the
absolute path of your current working directory with each `/`
replaced by `-`. For this repo cloned to e.g.
`/home/ubuntu/Python-forcefield-optimizer`, that's
`~/.claude/projects/-home-ubuntu-Python-forcefield-optimizer/`.

After `git clone`, the resume picker is empty for this project. Run:

```bash
bash scripts/install_resume_log.sh
```

from the repo root and the script:

1. Computes your machine's encoded-cwd path,
2. `mkdir -p`s it,
3. Gunzips each `transcripts/*.jsonl.gz` into it (skipping any
   filename that already exists — use `-f` to overwrite, or `-n`
   for a dry run).

After that, `claude --resume` (or `/resume` inside the CLI) lists the
archived sessions and you can pick one to continue. The transcript
file in `~/.claude/projects/` is independent of the gzipped one in
the repo — Claude Code appends new messages to *its* copy when you
resume, so the in-repo archive stays as a snapshot.

## Re-archiving

To save a *new* session into the repo:

```bash
# Find the live transcript Claude Code is writing
ls -lt ~/.claude/projects/-home-ubuntu-Python-forcefield-optimizer/*.jsonl

# Gzip the one you want and drop it in:
gzip -c <session-id>.jsonl > conversations/transcripts/<session-id>.jsonl.gz

# Commit and push.
```

(The current sessions IDs are UUIDs — not predictable, so use
`ls -lt` to pick the most recently-modified one.)
