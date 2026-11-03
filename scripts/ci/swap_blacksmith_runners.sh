#!/usr/bin/env bash
# Fork-only: rewrite upstream's paid runner labels/actions to GitHub-hosted ones.
#
# Upstream (NousResearch) runs CI on Blacksmith runners and GitHub "larger"
# runners that only exist in their org. In this fork those jobs sit queued
# forever, so CI never runs. Re-run this after every upstream sync merge:
#
#   scripts/ci/swap_blacksmith_runners.sh && git diff --stat .github
#
# Idempotent. tests.yml's slice count (8, not upstream's 2) and the test
# worker counts (4 in tests.yml and tests-os.yml's Windows lane) are separate
# hand edits; keep them when resolving those files.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
find .github -type f \( -name '*.yml' -o -name '*.yaml' \) -print0 | xargs -0 perl -pi -e '
  s/blacksmith-\d+vcpu-ubuntu-2404-arm/ubuntu-24.04-arm/g;
  s/blacksmith-\d+vcpu-ubuntu-2404/ubuntu-latest/g;
  s/blacksmith-\d+vcpu-windows-2025/windows-2025/g;
  s/blacksmith-\d+vcpu-macos-latest/macos-latest/g;
  s/windows-latest-32-arm-core/windows-11-arm/g;
  s/windows-latest-32-core/windows-latest/g;
  s{useblacksmith/checkout\@[0-9a-f]+\s*# v1}{actions/checkout\@f548e57e544e1ff5a4c46bf1e1b8685f8e4a348a # v7.0.1}g;
  s{useblacksmith/setup-docker-builder\@[0-9a-f]+\s*# v2}{docker/setup-buildx-action\@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f # v3}g;
  s{useblacksmith/build-push-action\@[0-9a-f]+\s*# v2}{docker/build-push-action\@v7}g;
'
if grep -rnE 'blacksmith-[0-9]+vcpu|useblacksmith/' .github --include='*.yml' --include='*.yaml'; then
  echo "unmapped Blacksmith reference left (above) — extend this script" >&2
  exit 1
fi
