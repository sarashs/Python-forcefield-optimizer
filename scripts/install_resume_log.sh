#!/usr/bin/env bash
# Install archived Claude Code session transcripts into the local
# `~/.claude/projects/<encoded-cwd>/` directory so that
# `claude --resume` (or `/resume` from inside the CLI) can find and
# replay them.
#
# Claude Code's resume picker enumerates `.jsonl` files in
# `~/.claude/projects/<encoded-cwd>/` where the encoded cwd is the
# current working directory with each `/` replaced by `-`. This
# script computes that path for whichever directory you run it from
# and gunzips the archived transcripts into it.
#
# Idempotent: re-running won't clobber an existing file (asks for -f
# to overwrite). Safe to run on any machine the repo gets cloned to.
#
# Usage (from the repo root):
#   bash scripts/install_resume_log.sh           # install all
#   bash scripts/install_resume_log.sh -f        # overwrite existing
#   bash scripts/install_resume_log.sh -n        # dry-run, print plan
set -euo pipefail

force=0
dry=0
while getopts "fnh" opt; do
    case "$opt" in
        f) force=1 ;;
        n) dry=1 ;;
        h)
            sed -n '2,/^set/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
src_dir="$repo_root/conversations/transcripts"
if [ ! -d "$src_dir" ]; then
    echo "error: $src_dir doesn't exist" >&2
    exit 1
fi

# claude-code encodes cwd by replacing / with -. So
# /home/ubuntu/Python-forcefield-optimizer becomes
# -home-ubuntu-Python-forcefield-optimizer.
cwd="$(pwd)"
encoded="${cwd//\//-}"
dest_dir="$HOME/.claude/projects/$encoded"

echo "repo:    $repo_root"
echo "cwd:     $cwd"
echo "encoded: $encoded"
echo "dest:    $dest_dir"
echo

if [ "$dry" -eq 0 ]; then
    mkdir -p "$dest_dir"
fi

count=0
for gz in "$src_dir"/*.jsonl.gz; do
    [ -e "$gz" ] || continue
    base="$(basename "$gz" .gz)"           # <session-id>.jsonl
    dest="$dest_dir/$base"
    if [ -e "$dest" ] && [ "$force" -eq 0 ]; then
        echo "skip (exists): $dest"
        echo "  → re-run with -f to overwrite"
        continue
    fi
    if [ "$dry" -eq 1 ]; then
        echo "would install: $gz  →  $dest"
    else
        gunzip -c "$gz" > "$dest"
        echo "installed: $dest"
    fi
    count=$((count + 1))
done

# Also copy any tool-result side directories if present
for sd in "$src_dir"/*/; do
    [ -e "$sd" ] || continue
    sid="$(basename "$sd")"
    dest="$dest_dir/$sid"
    if [ -e "$dest" ] && [ "$force" -eq 0 ]; then
        echo "skip (exists): $dest"
        continue
    fi
    if [ "$dry" -eq 1 ]; then
        echo "would copy:    $sd  →  $dest"
    else
        cp -r "$sd" "$dest"
        echo "copied:        $dest"
    fi
    count=$((count + 1))
done

echo
if [ "$dry" -eq 1 ]; then
    echo "dry-run — no files written. Re-run without -n to apply."
else
    echo "done — $count item(s) installed."
    echo "next: from this directory, run \`claude --resume\` (or just \`claude\`"
    echo "and use the resume picker). The session(s) above will be listed."
fi
