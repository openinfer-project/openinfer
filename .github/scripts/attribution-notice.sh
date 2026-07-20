#!/usr/bin/env bash
# A verified signature vouches for the author only when committer and author
# match (GitHub binds signatures to the committer) or the commit came from
# GitHub's web flow.
set -euo pipefail

: "${REPO:?}" "${PR_NUMBER:?}" "${PR_AUTHOR:?}"

export MARKER='<!-- commit-attribution-notice -->'

comment_id=$(gh api "repos/$REPO/issues/$PR_NUMBER/comments" --paginate \
  --jq '.[] | select((.body | startswith(env.MARKER))
                     and (.user.login // "") == "github-actions[bot]")
            | .id' | sed -n '1p')

upsert() {
  if [ -n "$comment_id" ]; then
    gh api --method PATCH "repos/$REPO/issues/comments/$comment_id" -f body="$1"
  else
    gh api "repos/$REPO/issues/$PR_NUMBER/comments" -f body="$1"
  fi
}

total=$(gh api "repos/$REPO/pulls/$PR_NUMBER" --jq .commits)
if [ "$total" -gt 250 ]; then
  upsert "$MARKER"$'\n'"### Commit attribution notice"$'\n\n'"This pull request has $total commits, more than the 250 the API exposes for scanning; commit attributions were not verified. This check never fails the build."
  exit 0
fi

suspicious=$(gh api "repos/$REPO/pulls/$PR_NUMBER/commits" --paginate --jq '
  .[]
  | select(.author != null and .author.login != env.PR_AUTHOR)
  | select(
      (.commit.verification.verified
       and ((.committer.login // "") == .author.login
            or (.committer.login // "") == "web-flow"))
      | not)
  | {sha: .sha[0:8], login: .author.login}')

if [ -z "$suspicious" ]; then
  if [ -n "$comment_id" ]; then
    upsert "$MARKER"$'\n'"All commit attributions previously flagged on this pull request are resolved."
  fi
  exit 0
fi

body=$(jq -rs --arg marker "$MARKER" --arg pr_author "$PR_AUTHOR" '
  map("- `" + .sha + "` attributed to @" + .login)
  | $marker + "\n### Commit attribution notice\n\n"
    + "This pull request contains commits attributed to an account other than "
    + "the PR author @" + $pr_author + " without a verified signature from that account:\n\n"
    + join("\n")
    + "\n\nIf you are mentioned above and did not take part in this pull request, "
    + "please say so here. Attributing a commit to someone else normally requires "
    + "that person to be involved; maintainers should treat unacknowledged "
    + "attribution as a review blocker.\n\nThis check never fails the build."' \
  <<<"$suspicious")

upsert "$body"
