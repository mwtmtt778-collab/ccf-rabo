#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  bash tools/git_checkpoint.sh \"commit message\""
    exit 1
fi

MSG="$1"

echo "========================================"
echo "Git checkpoint"
echo "========================================"

git status --short

echo
echo "Adding tracked/source changes..."
git add \
    .gitignore \
    agents \
    core \
    drivers \
    expert \
    tools \
    config \
    docs \
    requirements.txt \
    main.py \
    2>/dev/null || true

echo
echo "Staged files:"
git diff --cached --name-status

if git diff --cached --quiet; then
    echo
    echo "No source changes to commit."
    exit 0
fi

echo
git commit -m "$MSG"

echo
echo "Checkpoint created:"
git log -1 --oneline
