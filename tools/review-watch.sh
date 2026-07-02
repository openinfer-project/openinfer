#!/usr/bin/env bash
# AI-review watcher for our open openinfer PRs.
#
# Polls each PR for NEW review comments from bots (Codex) or the maintainer
# (xiaguan) since a saved baseline. When something new appears it does NOT
# auto-apply — it prints the new comments and exits non-zero so the caller
# (Claude) wakes up and runs a subagent to judge whether the comment is
# actually right before touching code. Priority when they conflict:
# xiaguan > Codex; both can be wrong, so the subagent verifies against the
# code, never rubber-stamps.
#
# Usage:
#   tools/review-watch.sh init            # snapshot current comment counts
#   tools/review-watch.sh check           # one-shot: exit 10 if new activity
#   tools/review-watch.sh watch [secs]    # loop until new activity (default 300s)

set -uo pipefail

REPO=openinfer-project/openinfer
PRS=(485 491)
STATE_DIR="${TMPDIR:-/tmp}/openinfer-review-watch"
mkdir -p "$STATE_DIR"

# "<bot-review-comments> <xiaguan-issue-comments>" fingerprint per PR.
fingerprint() {
  local pr=$1
  local bot xg
  bot=$(gh api "repos/$REPO/pulls/$pr/comments" \
    --jq '[.[] | select(.user.login != "n-WN")] | length' 2>/dev/null || echo "?")
  xg=$(gh pr view "$pr" -R "$REPO" --json comments \
    --jq '[.comments[] | select(.author.login == "xiaguan")] | length' 2>/dev/null || echo "?")
  echo "$bot $xg"
}

new_comments() {
  local pr=$1
  echo "=== PR #$pr new review activity ==="
  gh api "repos/$REPO/pulls/$pr/comments" \
    --jq '.[] | select(.user.login != "n-WN") | "[\(.user.login)] \(.path):\(.line)\n\(.body)\n"' 2>/dev/null
  gh pr view "$pr" -R "$REPO" --json comments \
    --jq '.comments[] | select(.author.login == "xiaguan") | "[xiaguan issue-comment] \(.body)\n"' 2>/dev/null
}

case "${1:-check}" in
  init)
    for pr in "${PRS[@]}"; do fingerprint "$pr" > "$STATE_DIR/pr-$pr"; done
    echo "baseline saved for PRs: ${PRS[*]}"
    ;;
  check)
    changed=0
    for pr in "${PRS[@]}"; do
      cur=$(fingerprint "$pr")
      old=$(cat "$STATE_DIR/pr-$pr" 2>/dev/null || echo "")
      if [ "$cur" != "$old" ]; then
        changed=1
        new_comments "$pr"
        echo "$cur" > "$STATE_DIR/pr-$pr"
      fi
    done
    [ "$changed" = 1 ] && exit 10 || echo "no new review activity"
    ;;
  watch)
    interval="${2:-300}"
    for pr in "${PRS[@]}"; do
      [ -f "$STATE_DIR/pr-$pr" ] || fingerprint "$pr" > "$STATE_DIR/pr-$pr"
    done
    while true; do
      for pr in "${PRS[@]}"; do
        cur=$(fingerprint "$pr")
        old=$(cat "$STATE_DIR/pr-$pr" 2>/dev/null || echo "")
        if [ "$cur" != "$old" ]; then
          new_comments "$pr"
          echo "$cur" > "$STATE_DIR/pr-$pr"
          exit 10
        fi
      done
      sleep "$interval"
    done
    ;;
  *)
    echo "usage: $0 {init|check|watch [secs]}" >&2
    exit 2
    ;;
esac
