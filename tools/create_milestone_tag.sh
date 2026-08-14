#!/usr/bin/env bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  bash tools/create_milestone_tag.sh TAG \"message\""
    exit 1
fi

TAG="$1"
MSG="$2"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree has source changes."
    echo "Commit them before creating a milestone tag."
    git status --short
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag already exists: $TAG"
    exit 1
fi

git tag -a "$TAG" -m "$MSG"

echo
echo "Created milestone tag:"
git show --no-patch --oneline "$TAG"

echo
echo "To push:"
echo "  git push github $TAG"
