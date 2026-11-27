"""Offline eval set for task routing. Every case is synthetic and hand-labelled.

Nothing here comes from real chats. Names, channels and numbers are invented.

A label says which tier the task NEEDS:

* ``premium`` — a cheaper model risks a visibly worse result: code, design,
  judgment, research synthesis, writing other people read, sensitive subjects,
  side effects, or anything whose meaning sits in earlier turns.
* ``cheap``   — a competent small model does it as well as the primary: read-only
  lookups, status checks, formatting and extraction of given text, summaries for
  the requester, board toggles, gathering raw material for a parent agent.

The set over-samples the hard edge on purpose: premium tasks dressed up as
mechanical ones ("list the reasons we should...", "summarise this and post it").
Those are the cases where a router loses quality.

Each case is ``(scope, label, text, extra)`` where ``extra`` may carry
``platform``, ``context`` (subagent), ``public`` or ``attachments``.
"""
from __future__ import annotations

P, C = "premium", "cheap"
TURN, SUB = "turn", "subagent"
CRON = {"platform": "cron"}

TURN_CASES: list[tuple[str, str, str, dict]] = [
    # ── cheap: read-only lookups and status ──
    (TURN, C, "grep for TODO in the repo", {}),
    (TURN, C, "list the files in plugins/", {}),
    (TURN, C, "show me the last 50 lines of the gateway log", {}),
    (TURN, C, "how many open issues are on the tracker?", {}),
    (TURN, C, "is the bestiary service up?", {}),
    (TURN, C, "check disk usage on the data volume", {}),
    (TURN, C, "what time is it in Wellington?", {}),
    (TURN, C, "fetch the latest release tag of the swarm-map repo", {}),
    (TURN, C, "can you list my cron jobs", {}),
    (TURN, C, "find every file under docs/ that mentions kanban", {}),
    (TURN, C, "count the lines in config.yaml", {}),
    (TURN, C, "please search my notes for the word ferment", {}),
    (TURN, C, "look up the weather for tomorrow", {}),
    (TURN, C, "cat the README in the tools folder", {}),
    (TURN, C, "[ari] list the open cards on the board", {}),
    (TURN, C, "status of the nightly backup?", {}),
    (TURN, C, "ping the camofox container", {}),
    (TURN, C, "diff config.yaml against the backup copy", {}),
    # ── cheap: formatting / extraction of supplied text ──
    (TURN, C, "convert this to json: name=Kiri, age=31, city=Napier", {}),
    (TURN, C, "sort these alphabetically: pear, apple, fig, kiwi", {}),
    (TURN, C, "extract all the URLs from the text below\nsee https://example.org and https://example.com/a", {}),
    (TURN, C, "dedupe this list: a, b, a, c, b", {}),
    (TURN, C, "reformat these dates as YYYY-MM-DD: 3 March 2026, 14/02/2026", {}),
    (TURN, C, "turn this into a markdown table\nfruit 3\nveg 5\nbread 1", {}),
    (TURN, C, "alphabetize the list I just pasted: zeta, alpha, mu", {}),
    # ── cheap: summaries for the requester ──
    (TURN, C, "summarize this log excerpt: 10:01 ok 10:02 ok 10:03 timeout 10:04 ok", {}),
    (TURN, C, "tl;dr of the pasted meeting notes please\nAgenda: snacks, rota. Decided: rota weekly.", {}),
    (TURN, C, "recap today's commits in the plume repo", {}),
    # ── cheap: board toggles, reminders, tests ──
    (TURN, C, "mark card 42 done", {}),
    (TURN, C, "run the tests", {}),
    (TURN, C, "remind me at 5pm to water the plants", {}),
    (TURN, C, "pin the message about the rota", {}),
    (TURN, C, "archive the done column", {}),
    (TURN, C, "tag card 17 as blocked", {}),
    # ── cheap: background jobs that only check and report ──
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response will be "
              "automatically delivered to the user.]\n\n[CURRENT RUN TIME: Monday 09:00]\n\n"
              "Check the RSS feeds in feeds.txt and list any new items since yesterday. "
              "If nothing is new, respond with [SILENT].", CRON),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\nCheck whether the five health "
              "URLs in monitors.md return 200 and list the ones that don't.", CRON),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\nSummarise the new entries in "
              "the shared inbox folder since the last run as bullet points.", CRON),
    (TURN, C, "run the nightly digest", CRON),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\nCount the new rows in the "
              "signups sheet and report the number.", CRON),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\n## Script Output\nThe following "
              "data was collected by a pre-run script. Use it as context for your analysis.\n\n"
              "```\ndisk 81%\nmem 40%\nload 0.7\n```\n\n"
              "List any metric above 80%.", CRON),

    # ── premium: code generation and debugging ──
    (TURN, P, "Write a Python function that merges two sorted linked lists.", {}),
    (TURN, P, "Implement the retry logic in agent/chat_completion_helpers.py and open a PR.", {}),
    (TURN, P, "fix the failing test in tests/tools", {}),
    (TURN, P, "grep the logs and tell me why the gateway crashed", {}),
    (TURN, P, "find the bug in parser.py", {}),
    (TURN, P, "run the tests and fix whatever fails", {}),
    (TURN, P, "check the PR for problems before I merge it", {}),
    (TURN, P, "add a --dry-run flag to the backup script", {}),
    (TURN, P, "refactor the routing module so it's easier to test", {}),
    (TURN, P, "write a regex that matches NZ phone numbers", {}),
    (TURN, P, "Traceback (most recent call last):\n  File \"x.py\", line 3\nKeyError: 'id'", {}),
    (TURN, P, "make a cron job that posts the weather every morning", {}),
    (TURN, P, "update the config to use the new model", {}),
    (TURN, P, "debug why the signal adapter drops voice notes", {}),
    # ── premium: architecture and diagnosis ──
    (TURN, P, "How should we structure the multi-tenant routing layer so tenants stay isolated?", {}),
    (TURN, P, "Design a system for multi-tenant routing across the fleet with failure isolation.", {}),
    (TURN, P, "Why is the deploy flaky lately?", {}),
    (TURN, P, "how many ways could this deadlock?", {}),
    (TURN, P, "what's the best architecture for this?", {}),
    (TURN, P, "Think through the tradeoffs between two providers and recommend one.", {}),
    (TURN, P, "what's the root cause of yesterday's outage?", {}),
    # ── premium: judgment, advice, planning ──
    (TURN, P, "should we ship this today?", {}),
    (TURN, P, "Not sure how to approach this — what do you think we should do about the parser?", {}),
    (TURN, P, "when is it worth adding a cache layer?", {}),
    (TURN, P, "whats your opinion on the design?", {}),
    (TURN, P, "what's the right way to handle auth here?", {}),
    (TURN, P, "list the reasons we should drop the old server", {}),
    (TURN, P, "prioritize the backlog for next week", {}),
    (TURN, P, "help me decide between the two flats", {}),
    (TURN, P, "plan my week around the three deadlines", {}),
    (TURN, P, "is it safe to delete the old volume?", {}),
    (TURN, P, "check if this clause is fair to us", {}),
    (TURN, P, "which option is cheaper long term?", {}),
    (TURN, P, "review my draft and tell me what's weak", {}),
    (TURN, P, "Can you look at the thing we discussed and get back to me?", {}),
    # ── premium: research and explanation ──
    (TURN, P, "research how other agent frameworks do model routing", {}),
    (TURN, P, "explain how prompt caching works", {}),
    (TURN, P, "compare glm and deepseek for tool calling", {}),
    (TURN, P, "analyze last month's spend and tell me where it went", {}),
    (TURN, P, "find out whether the library is still maintained", {}),
    (TURN, P, "what is a monad?", {}),
    (TURN, P, "tell me about the history of the treaty", {}),
    # ── premium: writing other people read ──
    (TURN, P, "draft an email to the landlord about the leak", {}),
    (TURN, P, "post the release notes to #announcements", {}),
    (TURN, P, "summarise this thread and send it to the team", {}),
    (TURN, P, "reply to Sam saying we can't make Friday", {}),
    (TURN, P, "write a tweet about the launch", {}),
    (TURN, P, "write the README section for the new flag", {}),
    (TURN, P, "proofread my cover letter", {}),
    (TURN, P, "announce the new rota in the group", {}),
    (TURN, P, "grep the announcement", {"public": True}),
    (TURN, P, "write a newsletter intro for December", {}),
    # ── premium: creative ──
    (TURN, P, "write a short poem about the garden", {}),
    (TURN, P, "brainstorm names for the new bot", {}),
    (TURN, P, "give me ideas for the zine cover", {}),
    # ── premium: sensitive ──
    (TURN, P, "how much tax do I owe on the contractor income?", {}),
    (TURN, P, "list my medications and the dosages", {}),
    (TURN, P, "rotate the API keys for the fleet", {}),
    (TURN, P, "check whether this login page is phishing", {}),
    (TURN, P, "translate this into te reo Māori", {}),
    (TURN, P, "i feel overwhelmed and can't start anything", {}),
    (TURN, P, "pay the invoice from the designer", {}),
    (TURN, P, "count how many days until my visa expires", {}),
    (TURN, P, "list the people who have access to the vault", {}),
    # ── premium: side effects ──
    (TURN, P, "delete the old backups", {}),
    (TURN, P, "restart the gateway", {}),
    (TURN, P, "send me the disk report every morning", {}),
    (TURN, P, "remove card 12", {}),
    (TURN, P, "rename the channel to garden-ops", {}),
    (TURN, P, "install ffmpeg in the container", {}),
    (TURN, P, "cancel my subscription to the newspaper", {}),
    # ── premium: context-dependent follow-ups ──
    (TURN, P, "yes", {}),
    (TURN, P, "do it", {}),
    (TURN, P, "same for the other two", {}),
    (TURN, P, "try again", {}),
    (TURN, P, "thanks!", {}),
    (TURN, P, "the second one", {}),
    # ── premium: uncertain / open ──
    (TURN, P, "", {}),
    (TURN, P, "hmm what about the other approach", {}),
    (TURN, P, "Sunday lunch plans?", {}),
    (TURN, P, "I've been thinking about the garden layout a lot", {}),
    (TURN, P, "list the items in this photo", {"attachments": True}),
    # ── premium: cron jobs that write or decide ──
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nWrite this week's community "
              "newsletter from the notes folder and post it to #general.", CRON),
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nReview the open PRs and "
              "recommend which to merge first.", CRON),
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nCheck the error logs and "
              "fix anything that's broken.", CRON),
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nRead today's inbox and "
              "reply to anything urgent.", CRON),
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nAnalyse the week's sales "
              "and explain what changed.", CRON),
]

SUBAGENT_CASES: list[tuple[str, str, str, dict]] = [
    # ── cheap: gathering, lookups, formatting for the parent ──
    (SUB, C, "Search the web for the current OpenRouter price of glm-5.3-flash and return the raw numbers "
             "with URLs.", {}),
    (SUB, C, "List every file under tools/ whose name starts with delegate_ and return the paths.", {}),
    (SUB, C, "Grep the repo for 'routing_override' and return each match with file and line.", {}),
    (SUB, C, "Fetch the three URLs in the context and return their page titles.",
     {"context": "https://example.org/a\nhttps://example.org/b\nhttps://example.org/c"}),
    (SUB, C, "Read docs/session-lifecycle.md and summarise it in ten bullet points.", {}),
    (SUB, C, "Count how many test files exist under tests/tools.", {}),
    (SUB, C, "Collect the last 20 lines of each log file in logs/ and return them.", {}),
    (SUB, C, "Gather the release dates of the last five tagged versions.", {}),
    (SUB, C, "Convert the CSV in the context to JSON.", {"context": "a,b\n1,2\n3,4"}),
    (SUB, C, "Extract every email address from the pasted text and return them as a list.",
     {"context": "Contact: someone@example.org, other@example.com"}),
    (SUB, C, "Summarize the contents of README.md for the parent agent.", {}),
    (SUB, C, "Check whether each of the ten links in links.txt returns HTTP 200 and report back.", {}),
    (SUB, C, "Look up the latest version of the requests package on PyPI.", {}),
    (SUB, C, "For each repo in repos.txt, return the default branch name.", {}),
    (SUB, C, "Run the test suite in tests/plugins/intelligent_routing and return the pass/fail counts.", {}),
    (SUB, C, "Scan the notes folder for any file mentioning 'compost' and list the paths.", {}),
    (SUB, C, "Tabulate the numbers in the context by month.", {"context": "Jan 3\nFeb 5\nMar 2"}),
    (SUB, C, "Sort the following items by date and return them.", {"context": "2026-03-01 x\n2026-01-02 y"}),
    (SUB, C, "Download the dataset at the given URL and return its row count.",
     {"context": "https://example.org/data.csv"}),
    (SUB, C, "Retrieve the page at the URL and return the text of the first heading.",
     {"context": "https://example.org"}),
    (SUB, C, "Recap the git log of the last 30 commits as one line each.", {}),
    (SUB, C, "Find all occurrences of 'TODO' in plugins/ and return file:line pairs.", {}),
    (SUB, C, "Inventory the docker images on this host and return name and size.", {}),
    (SUB, C, "Measure how long `ls -R docs` takes and report the time.", {}),
    (SUB, C, "Browse the changelog page and list entries from the last month.", {}),

    # ── premium: code generation and review ──
    (SUB, P, "Implement the escalation helper in agent/routing_override.py with tests.", {}),
    (SUB, P, "Review this diff for correctness bugs.", {"context": "diff --git a/x.py b/x.py\n+def f(): pass"}),
    (SUB, P, "Fix the flaky test in tests/tools/test_delegate_batch_tag.py.", {}),
    (SUB, P, "Write a migration script for the sessions table.", {}),
    (SUB, P, "Refactor delegate_tool.py so routing is a separate module.", {}),
    (SUB, P, "Find the bug that makes the child timeout fire early and propose a patch.", {}),
    (SUB, P, "Add a parser for the new config key and wire it into the loader.", {}),
    (SUB, P, "Debug the failing CI run and report the root cause.", {}),
    (SUB, P, "Write unit tests for the classifier.", {}),
    (SUB, P, "Build a small CLI tool that lists stale branches.", {}),
    # ── premium: architecture ──
    (SUB, P, "Design the data model for multi-tenant memory isolation.", {}),
    (SUB, P, "Assess whether the gateway can scale to 50 agents and where it would bottleneck.", {}),
    (SUB, P, "Investigate the race condition between the heartbeat and the stop fan-out.", {}),
    # ── premium: research synthesis ──
    (SUB, P, "Research how three other agent frameworks route between cheap and expensive models, and "
             "compare their approaches.", {}),
    (SUB, P, "Analyse the cost data in the context and explain what drives the spend.",
     {"context": "kimi 70%\nglm 5%\nother 25%"}),
    (SUB, P, "Evaluate the two libraries and recommend one.", {}),
    (SUB, P, "Explain how the credential pool picks a key.", {}),
    (SUB, P, "Fact-check the claims in the draft against primary sources.", {}),
    (SUB, P, "Write a short report on the state of local speech recognition for te reo Māori.", {}),
    # ── premium: public writing ──
    (SUB, P, "Draft the announcement post for the release.", {}),
    (SUB, P, "Write the PR description for these changes.", {}),
    (SUB, P, "Compose a reply to the customer's complaint.", {"context": "They were double charged."}),
    (SUB, P, "Rewrite the landing page copy to be clearer.", {}),
    (SUB, P, "Write release notes for v1.2.", {}),
    # ── premium: judgment ──
    (SUB, P, "Decide which of the open issues are worth fixing this week.", {}),
    (SUB, P, "Triage the new bug reports by severity.", {}),
    (SUB, P, "Critique the plan in the context and list its risks.", {"context": "Plan: migrate on Friday."}),
    (SUB, P, "Pick the best of the three candidate names.", {}),
    # ── premium: sensitive ──
    (SUB, P, "Audit the repo for leaked API keys.", {}),
    (SUB, P, "Check the contract in the context for unusual clauses.", {"context": "Clause 7: ..."}),
    (SUB, P, "Summarise the medical letter for the user.", {}),
    (SUB, P, "List which accounts have admin permissions.", {}),
    # ── premium: side effects ──
    (SUB, P, "Delete the stale branches older than six months.", {}),
    (SUB, P, "Restart every container that failed its health check.", {}),
    (SUB, P, "Send the weekly report to the three addresses in the context.", {"context": "a@example.org"}),
    (SUB, P, "Update the dependency pins and open a PR.", {}),
    (SUB, P, "Install the speakers model and configure it.", {}),
    # ── premium: cheap verb, premium content in context ──
    (SUB, P, "Summarise the context for the parent.",
     {"context": "Traceback (most recent call last):\n  File \"a.py\"\nValueError: bad"}),
    (SUB, P, "List the files.", {"context": "Then implement the missing handler in each one."}),
    (SUB, P, "Extract the key points.", {"context": "This is the draft we will publish on the blog tomorrow."}),
    (SUB, P, "Gather the transcripts.", {"context": "The recordings are in te reo Māori."}),
    # ── premium: uncertain ──
    (SUB, P, "Handle the rest of the migration work.", {}),
    (SUB, P, "Take care of it.", {}),
    (SUB, P, "Work out what's going on with the bot.", {}),
]

ALL_CASES = TURN_CASES + SUBAGENT_CASES
