---
name: project-doc
description: >
  Track any non-trivial task through a doc placed in the right domain directory
  under docs/ (models/<line>/, subsystems/<area>/, playbooks/, lessons/, etc.).
  Covers the full lifecycle: read docs first, write a plan, wait for approval,
  log execution, and capture lessons learned.
  Use this skill whenever the user gives a concrete task — not just Q&A or
  casual discussion. Even if the task seems straightforward, if it involves
  doing something (not just answering something), open or reuse a domain doc.
  Bias toward using this skill. When in doubt, use it.
---

# Project Doc

## Why This Exists

Every task is an opportunity to learn something or repeat a past mistake.
This skill ensures three things:

1. **You read before you act.** The repo has accumulated knowledge in docs/.
   Ignoring it means rediscovering problems that were already solved.
2. **The human approves before you execute.** This is an async workflow —
   the human needs to see and approve the plan before work begins.
3. **What happened gets recorded.** After execution, the doc captures what
   worked, what didn't, and what to remember next time.

## When To Use

Use this skill when the user gives a task that involves *doing* something —
writing code, running commands, debugging, deploying, refactoring, setting up,
fixing, creating, modifying, testing, or investigating.

Do not use it for:
- Pure questions ("what does this function do?")
- Casual discussion or brainstorming with no action item
- Continuing work on a task that already has an active project doc open in context

When in doubt, use it. An unnecessary project doc costs a minute.
A missed one costs repeated mistakes.

## Lifecycle

```
Prepare → Review Gate → Execute → Debrief
```

You must not skip phases or combine them. Each phase ends with a clear
handoff to the user.

---

## Phase 1: Prepare

### 1.1 Read the index

Read `docs/index.md`. Every time, no exceptions. Use it to identify which
other documents are relevant to this task.

### 1.2 Read relevant docs

Based on the index, read the minimal set of documents that could inform
this task. Think about:

- Is there a doc about the model line or subsystem being touched
  (`docs/models/<line>/`, `docs/subsystems/<area>/`)?
- Is there a relevant playbook or lesson (`docs/playbooks/`, `docs/lessons/`)?
- Is there a past task doc on similar work in the same domain directory?
- If writing code or scripts, are there existing implementations to learn from?

Be honest about what you read. Only list files you actually opened.

### 1.3 Find or create the task doc

Decide where the doc belongs using the classification rule in `CLAUDE.md`
(model line / subsystem / playbook / lesson / benchmark / roadmap). If a doc
in that directory already covers this task, reuse it.

- Exists and active → reuse it, update the Preparation section.
- Nothing matches → create a new file with a descriptive slug in the
  appropriate directory.
  Examples:
    - `docs/models/qwen35/fix-decode-state-leak.md`
    - `docs/subsystems/scheduler/chunked-prefill.md`
    - `docs/playbooks/nsys-trace-collection.md`

### 1.4 Write the Preparation section

```markdown
## Preparation

- **Read**:
  - `path/to/doc` — what it told us
  - `path/to/doc` — what it told us
- **Relevant history**:
  - `docs/<domain>/past-task.md` — learned that X causes Y
  - (or: no relevant past docs found)
- **Plan**:
  1. Concrete step with specifics (files, commands, resources)
  2. Concrete step
  3. Concrete step
- **Risks / open questions**:
  - What could go wrong or what's still unknown
  - (omit section if genuinely none)
```

Rules:
- **Read** — only files you actually opened. No padding.
- **Relevant history** — past project docs, known pitfalls, lessons learned.
  This is how you avoid repeating mistakes. Look for them.
- **Plan** — concrete and specific. "Investigate the issue" is not a plan step.
  *How* will you investigate? What will you look at? What commands will you run?
- **Risks** — if something might go wrong or you're unsure about something,
  say so now. It's much cheaper to catch it here than during execution.

### 1.5 Stop and ask for review

Present a short summary and explicitly ask the user to approve before
you do anything.

Format:

```
📋 Task doc: `docs/<domain>/<name>.md`

Read:
- `docs/index.md`
- `<other files>`

Plan:
1. ...
2. ...

Risks: <one-liner or "none identified">

Please review before I start.
```

**Do not proceed until the user explicitly approves.**

---

## Phase 2: Execute

Once approved, do the work. As you go, maintain an execution log in
the project doc:

```markdown
## Execution Log

### Step 1: <what you planned>
- What you actually did
- Commands run, files modified
- Result: success / partial / failed

### Step 2: ...

### Unexpected
- Anything that surprised you or deviated from the plan
- Errors encountered and how you resolved them
- Things that were harder or easier than expected
```

Rules:
- Log as you go, not after the fact.
- Be specific: include actual error messages, actual file paths,
  actual command output (abbreviated if very long).
- If you deviate from the plan, note why.
- If you hit a problem that blocks progress, stop and ask the user
  rather than guessing.

---

## Phase 3: Debrief

After execution is complete (or if you're blocked and stopping), write
the debrief section:

```markdown
## Debrief

- **Outcome**: What was accomplished (or not)
- **Pitfalls encountered**:
  - What went wrong or was unexpectedly tricky
  - Root cause if known
- **Lessons learned**:
  - What should be remembered for next time
  - What docs should be updated
- **Follow-ups**:
  - Any remaining work or related tasks
  - (omit if none)
```

This is the most important section for long-term value. A project doc
without a debrief is just a to-do list. The debrief is what prevents
the same mistake from happening twice.

Present the debrief to the user as a summary when the task is done.

---

## File Template

When creating a new task doc, use this structure:

```markdown
# <Task Title>

> **TL;DR:** One-line summary. Keep it true as the task progresses — that's
> the only thing readers can rely on.
>
> **Last touched:** YYYY-MM   <!-- only for docs/models/ and docs/subsystems/ -->

## Preparation

(filled in Phase 1)

## Execution Log

(filled in Phase 2)

## Debrief

(filled in Phase 3)
```

Update the **TL;DR** whenever the headline state of the work changes. Bump
**Last touched** when you do real work on the doc (skip it for typo fixes).
Add a new row to `docs/index.md` when creating the doc; only update an
existing row if its TL;DR has become actively misleading.

---

## Principles

- **Read first, act second.** The cost of reading a doc is minutes.
  The cost of not reading it is hours of debugging something already solved.
- **Be honest about what you don't know.** Flagging uncertainty in
  Preparation is a feature, not a weakness.
- **Log the ugly parts.** The errors, the wrong turns, the surprises —
  these are the most valuable things to record.
- **The debrief is not optional.** Even for tasks that went smoothly,
  write one. "Everything went as planned, no issues" is a valid debrief.
  But write it.
