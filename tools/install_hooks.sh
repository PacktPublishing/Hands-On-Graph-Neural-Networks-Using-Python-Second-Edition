#!/bin/sh
# Install the repository's git hooks into .git/hooks.
#
#     sh tools/install_hooks.sh
#
# Hooks live in tools/hooks/ so they are versioned with the book; .git/hooks is
# local to each clone, so every clone runs this once.

set -e
root=$(git rev-parse --show-toplevel)
for hook in "$root"/tools/hooks/*; do
    name=$(basename "$hook")
    cp "$hook" "$root/.git/hooks/$name"
    chmod +x "$root/.git/hooks/$name"
    echo "installato  .git/hooks/$name"
done
