#!/usr/bin/env bash
set -euo pipefail

# Create target directories if they don't exist
mkdir -p linear log

# Move linear-scale plots
for img in w_*_linear.jpg; do
  mv -- "$img" linear/
done

# Move log-scale plots
for img in w_*_log.jpg; do
  mv -- "$img" log/
done

echo "✔️  Moved all linear plots to ./linear and all log plots to ./log"
