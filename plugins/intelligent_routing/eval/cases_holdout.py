"""Held-out eval set: written after the router was tuned on ``cases.py``, scored once.

Same format and labelling rules as ``cases.py``. The router's keyword lists were
never adjusted against these cases before the first scored run; the numbers that
run produced are recorded in the README. Synthetic, invented content only.
"""
from __future__ import annotations

from plugins.intelligent_routing.eval.cases import C, CRON, P, SUB, TURN

HOLDOUT_TURN: list[tuple[str, str, str, dict]] = [
    # cheap
    (TURN, C, "show the open pull requests count", {}),
    (TURN, C, "what's the uptime on the mini?", {}),
    (TURN, C, "list everything in the downloads folder", {}),
    (TURN, C, "get me the current exchange rate NZD to AUD", {}),
    (TURN, C, "search the wiki for 'compost rota'", {}),
    (TURN, C, "find the config file for the ops agent", {}),
    (TURN, C, "check if port 8787 is listening", {}),
    (TURN, C, "print the contents of notes/todo.md", {}),
    (TURN, C, "extract the phone numbers from this: call 021 555 0101 or 09 555 0199", {}),
    (TURN, C, "format this as a bullet list: eggs, flour, milk", {}),
    (TURN, C, "convert 3pm NZDT to UTC", {}),
    (TURN, C, "sort these by size: 3MB, 1GB, 20KB", {}),
    (TURN, C, "summarise the pasted changelog\n- add x\n- fix y\n- drop z", {}),
    (TURN, C, "mark the shopping card as done", {}),
    (TURN, C, "label card 9 urgent", {}),
    (TURN, C, "run the linter", {}),
    (TURN, C, "remind me tomorrow at 9 to call the vet", {}),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\nList new issues opened in the "
              "last 24 hours on the three repos in repos.txt.", CRON),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\nCheck the SSL expiry date of "
              "each domain in domains.txt and list any expiring within 14 days.", CRON),
    (TURN, C, "[IMPORTANT: You are running as a scheduled cron job.]\n\nDigest of new posts in the "
              "saved feeds, as bullet points.", CRON),
    # premium
    (TURN, P, "write a bash script to rotate the logs", {}),
    (TURN, P, "the build is failing, can you sort it out", {}),
    (TURN, P, "check the code in utils.py and clean it up", {}),
    (TURN, P, "list what's wrong with this function\ndef f(x): return x/0", {}),
    (TURN, P, "is this a good idea or am I overthinking it", {}),
    (TURN, P, "summarise the doc and tell me if we should sign it", {}),
    (TURN, P, "what are the downsides of moving to postgres", {}),
    (TURN, P, "find me the cheapest flight to Sydney and book it", {}),
    (TURN, P, "reply to the group with the new times", {}),
    (TURN, P, "make the bio on the website sound warmer", {}),
    (TURN, P, "count the votes and announce the winner", {}),
    (TURN, P, "list the steps to set up the new laptop", {}),
    (TURN, P, "check my maths on the quote: 3 days at 800 is 2600?", {}),
    (TURN, P, "how do I get the agents to stop replying to each other", {}),
    (TURN, P, "grep for the password in the env file", {}),
    (TURN, P, "fetch the transcript and translate it to English", {}),
    (TURN, P, "look at the screenshot and tell me what's broken", {"attachments": True}),
    (TURN, P, "tell the bot to be nicer to newcomers", {}),
    (TURN, P, "ok now the other repo", {}),
    (TURN, P, "go ahead", {}),
    (TURN, P, "what did you mean by that", {}),
    (TURN, P, "write up what we learned this week", {}),
    (TURN, P, "outline a talk about the garden for the meetup", {}),
    (TURN, P, "check whether the new tenant agreement is legal", {}),
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nDraft a friendly weekly update "
              "for the community from the notes and deliver it.", CRON),
    (TURN, P, "[IMPORTANT: You are running as a scheduled cron job.]\n\nLook at yesterday's errors and "
              "decide whether anyone needs to be woken up.", CRON),
]

HOLDOUT_SUB: list[tuple[str, str, str, dict]] = [
    # cheap
    (SUB, C, "List the functions defined in tools/delegate_tool_routes.py and return their names.", {}),
    (SUB, C, "Fetch the README of each repo in the context and return the first paragraph.",
     {"context": "repo-a\nrepo-b"}),
    (SUB, C, "Search the web for the release date of Python 3.14 and return the source URL.", {}),
    (SUB, C, "Count the lines of code in plugins/intelligent_routing.", {}),
    (SUB, C, "Return the raw output of `git log --oneline -20`.", {}),
    (SUB, C, "Collect every config key under 'routing:' mentioned in the docs folder.", {}),
    (SUB, C, "Extract the dates and amounts from the table in the context as CSV.",
     {"context": "| 2026-01-01 | 40 |\n| 2026-02-01 | 55 |"}),
    (SUB, C, "Summarise the five newest files in notes/ for the parent, one line each.", {}),
    (SUB, C, "Check each URL in the context and report which ones redirect.",
     {"context": "https://example.org\nhttps://example.net"}),
    (SUB, C, "Gather the open issue titles from the tracker into a list.", {}),
    # premium
    (SUB, P, "Write the code for a retry decorator with jitter.", {}),
    (SUB, P, "Go through the failing tests and make them pass.", {}),
    (SUB, P, "Summarise the security advisory and say whether we are affected.", {}),
    (SUB, P, "List the risks in moving the fleet to a new host.", {}),
    (SUB, P, "Read the three proposals and rank them.", {}),
    (SUB, P, "Draft a polite message declining the invitation.", {}),
    (SUB, P, "Search for prior art and explain how it differs from our approach.", {}),
    (SUB, P, "Rename the variables in parser.py to be clearer.", {}),
    (SUB, P, "Collect the logs and work out why the job died.", {}),
    (SUB, P, "Fetch the user's bank statements and categorise the spending.", {}),
    (SUB, P, "Check the translation of the karakia for accuracy.", {}),
    (SUB, P, "Do whatever the parent needs here.", {}),
]

HOLDOUT_CASES = HOLDOUT_TURN + HOLDOUT_SUB
