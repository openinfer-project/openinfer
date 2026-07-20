#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" || exit 1

WRITES=$(mktemp)
trap 'rm -f "$WRITES"' EXIT
FAILED=0

# shellcheck disable=SC2317,SC2329
gh() {
  local a prev="" jq_prog="" url=""
  for a in "$@"; do
    if [ "$a" = "--method" ] || [ "$a" = "-f" ]; then
      printf '%s\n' "gh $*" >>"$WRITES"
      return 0
    fi
    if [ "$prev" = "--jq" ]; then
      jq_prog="$a"
    fi
    case "$a" in repos/*) url="$a" ;; esac
    prev="$a"
  done
  local data='[]'
  case "$url" in
    */pulls/*/commits) data="$COMMITS_FIXTURE" ;;
    */issues/*/comments) data="$COMMENTS_FIXTURE" ;;
    */pulls/*) data="$PR_FIXTURE" ;;
  esac
  jq -cr "$jq_prog" <<<"$data"
}

expect() {
  if grep -qF -- "$2" "$WRITES"; then
    echo "  ok: $1"
  else
    echo "  FAIL: $1"
    FAILED=1
  fi
}

expect_absent() {
  if grep -qF -- "$2" "$WRITES"; then
    echo "  FAIL: $1"
    FAILED=1
  else
    echo "  ok: $1"
  fi
}

expect_no_writes() {
  if [ -s "$WRITES" ]; then
    echo "  FAIL: expected no writes, got: $(cat "$WRITES")"
    FAILED=1
  else
    echo "  ok: no writes"
  fi
}

run_script() {
  : >"$WRITES"
  local rc=0
  set +e
  # shellcheck disable=SC1091
  (. ./attribution-notice.sh)
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: script exited nonzero"
    FAILED=1
  fi
}

export REPO=test/repo PR_NUMBER=1 PR_AUTHOR=alice
export PR_FIXTURE='{"commits":5}'

COMMITS_MIXED='[
  {"sha":"aaaa111100000000","author":{"login":"alice"},"committer":{"login":"alice"},"commit":{"verification":{"verified":false}}},
  {"sha":"bbbb111100000000","author":{"login":"bob"},"committer":{"login":"bob"},"commit":{"verification":{"verified":false}}},
  {"sha":"bbbb222200000000","author":{"login":"bob"},"committer":{"login":"bob"},"commit":{"verification":{"verified":false}}},
  {"sha":"cccc111100000000","author":null,"committer":null,"commit":{"verification":{"verified":false}}},
  {"sha":"dddd111100000000","author":{"login":"bob"},"committer":{"login":"bob"},"commit":{"verification":{"verified":true}}},
  {"sha":"eeee111100000000","author":{"login":"bob"},"committer":{"login":"eve"},"commit":{"verification":{"verified":true}}},
  {"sha":"ffff111100000000","author":{"login":"carol"},"committer":{"login":"web-flow"},"commit":{"verification":{"verified":true}}}
]'
COMMITS_CLEAN='[
  {"sha":"aaaa111100000000","author":{"login":"alice"},"committer":{"login":"alice"},"commit":{"verification":{"verified":false}}},
  {"sha":"dddd111100000000","author":{"login":"bob"},"committer":{"login":"bob"},"commit":{"verification":{"verified":true}}}
]'
COMMENTS_NONE='[]'
COMMENTS_MARKED='[{"id":42,"user":{"login":"github-actions[bot]"},"body":"<!-- commit-attribution-notice -->\nold"}]'
COMMENTS_SQUATTED='[{"id":7,"user":{"login":"mallory"},"body":"<!-- commit-attribution-notice -->\nfake"}]'

echo "scenario: cross-author commits without the author's own signature are flagged"
COMMITS_FIXTURE=$COMMITS_MIXED COMMENTS_FIXTURE=$COMMENTS_NONE run_script
expect "creates a new comment" "repos/test/repo/issues/1/comments -f"
expect "flags bob's first unsigned commit" "bbbb1111"
expect "flags bob's second unsigned commit" "bbbb2222"
expect "flags verified commit signed by a different committer" "eeee1111"
expect_absent "PR author's own unsigned commit exempt" "aaaa1111"
expect_absent "unlinked-email commit exempt" "cccc1111"
expect_absent "self-signed verified commit exempt" "dddd1111"
expect_absent "web-flow commit exempt" "ffff1111"
expect_absent "does not patch when no comment exists" "--method PATCH"

echo "scenario: clean PR produces no writes"
COMMITS_FIXTURE=$COMMITS_CLEAN COMMENTS_FIXTURE=$COMMENTS_NONE run_script
expect_no_writes

echo "scenario: clean PR with stale notice patches it to resolved"
COMMITS_FIXTURE=$COMMITS_CLEAN COMMENTS_FIXTURE=$COMMENTS_MARKED run_script
expect "patches the existing comment" "--method PATCH repos/test/repo/issues/comments/42"
expect "patched body is the resolved notice" "are resolved"
expect_absent "does not create a second comment" "issues/1/comments -f"

echo "scenario: flagged PR with existing notice updates it in place"
COMMITS_FIXTURE=$COMMITS_MIXED COMMENTS_FIXTURE=$COMMENTS_MARKED run_script
expect "patches the existing comment" "--method PATCH repos/test/repo/issues/comments/42"
expect "patched body carries the flagged sha" "bbbb1111"
expect_absent "does not create a second comment" "issues/1/comments -f"

echo "scenario: marker comment from a non-bot author is ignored"
COMMITS_FIXTURE=$COMMITS_MIXED COMMENTS_FIXTURE=$COMMENTS_SQUATTED run_script
expect "creates its own comment" "issues/1/comments -f"
expect_absent "does not touch the squatted comment" "comments/7"

echo "scenario: PR beyond the 250-commit window is reported, never resolved"
COMMITS_FIXTURE=$COMMITS_CLEAN COMMENTS_FIXTURE=$COMMENTS_MARKED PR_FIXTURE='{"commits":300}' run_script
expect "patches the existing comment" "--method PATCH repos/test/repo/issues/comments/42"
expect "reports the unscannable commit count" "300 commits"
expect_absent "does not claim attributions resolved" "are resolved"

if [ "$FAILED" -eq 0 ]; then
  echo "all scenarios passed"
fi
exit "$FAILED"
