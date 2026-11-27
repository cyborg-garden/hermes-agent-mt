"""Task-type routing: decide cheap or premium from what a task is, not how long it is.

One pure function, :func:`classify_task`, serves both scopes:

* ``turn``     — the main chat turn (the ``pre_llm_call`` hook).
* ``subagent`` — one ``delegate_task`` child (the ``task-routing`` delegation backend).

The rule is asymmetric on purpose. Premium is the default. A task reaches the cheap
tier only when it carries POSITIVE evidence of cheap-safe work (a read-only lookup,
formatting, extraction, summarising, a status check, a board toggle, gathering for a
parent) AND none of the premium signals fire anywhere in its text. A premium signal
is any sign of code generation, design, judgment, research synthesis, writing for
other people, sensitive subject matter, side effects that are hard to undo, or a
reply that leans on earlier context. A false premium costs money; a false cheap costs
quality, so the premium lists are broad and the cheap lists are narrow.

Length is not a signal. A long cron prompt that only checks feeds is cheap; a short
"should we ship this?" is premium.

Everything here is deterministic and does no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ── Tiers ──
TIER_CHEAP = "cheap"
TIER_PREMIUM = "premium"

# ── Task types ──
# Cheap-safe types.
T_MECHANICAL = "mechanical"        # board toggles, reminders, run the tests
T_LOOKUP = "lookup"                # read-only fetch / list / search / status
T_FORMATTING = "formatting"        # convert, reformat, extract, sort given text
T_SUMMARISE = "summarise"          # summarise / recap / digest, for the requester
T_ORCHESTRATION = "orchestration"  # gather / collect / run-and-report for a parent
CHEAP_TYPES = frozenset({T_MECHANICAL, T_LOOKUP, T_FORMATTING, T_SUMMARISE, T_ORCHESTRATION})

# Premium types.
T_CODE_GEN = "code_gen"
T_ARCHITECTURE = "architecture"
T_RESEARCH = "research"
T_PUBLIC_WRITING = "public_writing"
T_CREATIVE = "creative"
T_JUDGMENT = "judgment"
T_SENSITIVE = "sensitive"
T_SIDE_EFFECT = "side_effect"
T_CONTEXTUAL = "contextual"
T_PROTECTED = "protected"
T_MULTIMODAL = "multimodal"
T_UNCERTAIN = "uncertain"
PREMIUM_TYPES = frozenset({
    T_CODE_GEN, T_ARCHITECTURE, T_RESEARCH, T_PUBLIC_WRITING, T_CREATIVE, T_JUDGMENT,
    T_SENSITIVE, T_SIDE_EFFECT, T_CONTEXTUAL, T_PROTECTED, T_MULTIMODAL, T_UNCERTAIN,
})
ALL_TYPES = CHEAP_TYPES | PREMIUM_TYPES

SCOPE_TURN = "turn"
SCOPE_SUBAGENT = "subagent"

# Above this many characters the text is too big to judge from keywords with any
# confidence (pasted documents, transcripts): premium.
_MAX_JUDGEABLE_CHARS = 12000


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# ── Premium signals, checked over the whole text (instructions AND context) ──
# Order is the precedence used for the reported type when several fire.
_PREMIUM_SIGNALS: tuple[tuple[str, re.Pattern], ...] = (
    (T_SENSITIVE, _rx(
        r"\b(legal|lawyer|attorney|lawsuit|sue|contract|nda|liabilit\w*|compliance|gdpr|"
        r"tax(es)?|irs|ird|invoice|payroll|salar(y|ies)|wire|bank\w*|payment|pay (him|her|them|\w+ for)|refund|"
        r"invest\w*|stock|crypto|loan|mortgage|budget)\b",
        r"\b(medical|medicine|doctors?|diagnos\w*|symptom\w*|medications?|dosages?|doses?|prescri\w*|"
        r"therap\w*|mental health|suicid\w*|self[- ]harm|grief|pregnan\w*)\b",
        r"\b(password|passphrase|credential\w*|secret\w*|api[ _-]?keys?|private key|2fa|mfa|ssh key|"
        r"vulnerab\w*|exploit\w*|cve-\d|malware|phishing|security|permissions?|access|admins?)\b",
        r"\b(harass\w*|abuse\w*|stalk\w*|threat\w*|doxx\w*|consent|minor|child(ren)?|immigration|visa)\b",
        r"\b(te reo|m[aā]ori|tikanga|kaitiakitanga|taonga|iwi|hap[uū]|marae|whakapapa|karakia|waiata|"
        r"t[aā] moko|translat\w*)\b",
        r"\b(i feel|i'm (so )?(sad|anxious|stressed|upset|worried|scared|depressed|overwhelmed|lonely|angry)|"
        r"i am (so )?(sad|anxious|stressed|upset|worried|scared|depressed|overwhelmed|lonely|angry))\b",
    )),
    (T_PUBLIC_WRITING, _rx(
        r"\b(publish\w*|announce\w*|announcement|tweet\w*|toot|blog|newsletter|press release|"
        r"op-?ed|linkedin|instagram|tiktok|youtube|caption|landing page|website copy|marketing|"
        r"pitch|outreach|cold email|grant application|cover letter|resume|cv|bio|proposal|talk|presentation|"
        r"slides?|speech|keynote|panel)\b",
        r"\bpost(ing|ed)?\b(?! ?request)(?!gres)",
        r"\b(send|email|message|dm|reply|respond|write back)\b[^.\n]{0,40}\b(to|back to)\b\s+(?!me\b|myself\b)\w+",
        r"\b(draft|write|compose|word|rewrite|edit|proofread|polish)\b[^.\n]{0,30}"
        r"\b(email|letter|message|reply|response|post|thread|article|essay|announcement|copy|"
        r"statement|note to|readme|docs?|documentation|changelog|release notes|description|summary for)\b",
        r"\bon behalf of\b",
        r"\b(to|for|with) the (team|channel|group|server|community|public|client|customer|users|board|investors?)\b",
        r"\bpublicly\b",
        r"#[a-z][\w-]+",  # a named channel destination
    )),
    (T_CREATIVE, _rx(
        r"\b(poem|story|stories|song|lyrics?|joke|fiction|novel|screenplay|haiku|limerick|"
        r"creative|brainstorm\w*|slogan|tagline|names? for|ideas? for|imagine|roleplay)\b",
    )),
    (T_CODE_GEN, _rx(
        r"\b(implement\w*|refactor\w*|debug\w*|patch\w*|hotfix|bugfix|regression|"
        r"pull request|\bpr\b|code review|unit tests?|write tests?|test case|"
        r"commit|merge|rebase|cherry-?pick|deploy\w*|migrat\w*|schema|sql|dockerfile|"
        r"config(uration)? change|code|clean (it|this|that|them) up|tidy (it|this|the code) up)\b",
        r"\bfix(es|ed|ing)?\b",
        r"\b(write|add|create|build|make|generate)\b[^.\n]{0,30}\b(code|program|tool|bot|plugin|"
        r"feature|cli|app|website|page|component|handler|hook|tests?|job|cron|workflow|pipeline|"
        r"function|class|method|module|script|query|regex|endpoint|api|parser)\b",
        r"```", r"\bdef \w+\(", r"\bclass \w+[(:]", r"traceback \(most recent", r"\bstack ?trace\b",
        r"\b(npm|pip|cargo|make|pytest|go) (install|build|run|test)\b",
    )),
    (T_ARCHITECTURE, _rx(
        r"\b(architect\w*|system design|design (a|an|the|our)|data model|threat model|"
        r"scal(e|ing|ability)|trade-?offs?|bottleneck\w*|race condition|deadlock\w*|concurren\w*|bugs?|"
        r"crash\w*|broken|why|"
        r"root cause|post-?mortem|incident|outage|why (is|does|did|are|was|do|has|won't|isn't|doesn't)|"
        r"how (should|would|could|do) (we|i|you|it)|structure (this|the|our)|approach (this|to))\b",
    )),
    (T_RESEARCH, _rx(
        r"\b(research\w*|investigat\w*|analy[sz]\w*|assess\w*|evaluat\w*|deep[- ]dive|synthes\w*|"
        r"conclu\w*|interpret\w*|agree on|disagree|"
        r"literature|sources|cite|citations?|fact[- ]?check\w*|verify (whether|that|if)|"
        r"find out|look into|dig into|explain\w*|teach|tutorial|compar\w*|vs\.?|versus|"
        r"differences? between|history of|overview of|write-?up|the impact|implications?)\b",
    )),
    (T_JUDGMENT, _rx(
        r"\b(should|shall (i|we)|ought|rank\w*|rate (them|these|it|each|this)|rating|grade|score (them|these|each)|cheapest|fastest|"
        r"math(s|ematics)?|calculat\w*|correct|accura\w*|right\?|is (this|that|it) right|what do you think|thoughts\??|opinion|advice|advise|"
        r"recommend\w*|suggest\w*|decide|deciding|decisions?|worth|best|right way|better|ideal|"
        r"pros and cons|critique|feedback|review|prioriti[sz]\w*|triage|plan(s|ning|ned)?|strateg\w*|roadmap|"
        r"think (through|about)|figure out|reason about|help me (decide|choose|understand|think)|"
        r"choose|pick (a|the|one|between)|which (one|is|should|option)|negotiat\w*|"
        r"persuad\w*|convince|apolog\w*|sorry|safe to|risky|risk\w*|is it ok|okay to|"
        r"wise|careful|sensible|fair)\b",
    )),
    (T_SIDE_EFFECT, _rx(
        r"\b(delete|remove|drop|purge|wipe|erase|destroy|kill|terminate|uninstall|revoke|"
        r"force[- ]push|reset --hard|rm -rf|rm\b|restart|reboot|shut ?down|stop|start (the|a|an)|"
        r"deploy|rollback|roll back|install|upgrade|downgrade|rotate|"
        r"buy|purchase|place an order|book|booking|reserve|cancel|unsubscribe|transfer|pay|send|forward|invite|ban|kick|"
        r"edit|modify|change|update|write|create|configure|set up|setup|enable|disable|"
        r"rename|move|overwrite|replace|approve|reject|grant)\b",
    )),
)

# Short follow-ups that only make sense against earlier turns: the answer quality
# depends on whatever came before, which we cannot see from the text.
_CONTEXTUAL_RX = _rx(
    r"^(yes|yeah|yep|yup|ok|okay|k|sure|go|go ahead|do it|continue|carry on|keep going|proceed|"
    r"again|retry|try again|same|lgtm|sounds good|perfect|great|thanks?|thank you|ty|cool|nice|no|nope|"
    r"wait|hmm+|huh|what|why|and\??|so\??|ok so|right)\W*$",
    r"\b(same (for|with|as)|the (other|previous|last|first|second|third) (one|ones|bit|part|option)\b|"
    r"the (rest|others)\b|"
    r"that one|this one|like before|as before|as discussed|we discussed|we talked|you said|you mentioned|"
    r"earlier|(the|see|as) above|above (message|text|thread|list)|the thing|get back to me|when you get a chance|whenever you)\b",
)

# ── Cheap evidence (positive signal), checked on the INSTRUCTION text only ──
# Leading verbs that mark read-only or trivially reversible work. Side-effect verbs
# (send/delete/restart/update/...) are deliberately absent: a misfire there acts on
# the world, and the premium side-effect list above vetoes them anyway.
_CHEAP_LEAD_VERBS: dict[str, str] = {
    **dict.fromkeys(("list", "show", "get", "fetch", "grep", "find", "search", "cat", "ls", "print",
                     "count", "check", "look", "lookup", "read", "open", "ping", "diff", "tail",
                     "head", "query", "browse", "scrape", "crawl", "download", "pull", "retrieve",
                     "locate", "status", "monitor", "poll", "watch", "measure"), T_LOOKUP),
    **dict.fromkeys(("convert", "reformat", "format", "extract", "parse", "sort", "dedupe",
                     "deduplicate", "alphabetize", "alphabetise", "tabulate", "normalize",
                     "normalise", "transform", "split", "join", "merge-sort", "strip", "trim",
                     "capitalize", "lowercase", "uppercase", "wrap"), T_FORMATTING),
    **dict.fromkeys(("summarize", "summarise", "recap", "tldr", "tl;dr", "condense", "digest",
                     "shorten", "skim"), T_SUMMARISE),
    **dict.fromkeys(("mark", "tag", "label", "pin", "unpin", "archive", "unarchive", "star",
                     "remind", "snooze", "close", "run", "rerun"), T_MECHANICAL),
    **dict.fromkeys(("gather", "collect", "compile", "aggregate", "enumerate", "inventory",
                     "catalog", "catalogue", "index", "map", "scan", "sweep", "audit-list",
                     "tally"), T_ORCHESTRATION),
}

# Phrases that carry cheap intent anywhere in the instruction (not only leading).
_CHEAP_PHRASES: tuple[tuple[str, re.Pattern], ...] = (
    (T_SUMMARISE, _rx(r"\b(summary of|summari[sz]e|recap|tl;?dr|digest|key points|bullet points? of|"
                      r"what('s| is) new|anything new|new (posts|items|entries|messages|issues))\b")),
    (T_FORMATTING, _rx(r"\b(as (json|csv|yaml|a table|markdown|a list|bullets)|"
                       r"into (an? )?(markdown |html |plain )?(json|csv|yaml|table|markdown|list|bullets|bullet points)|"
                       r"to (json|csv|yaml|markdown)|in (json|csv|yaml) format)\b")),
    (T_LOOKUP, _rx(r"\b(how many|what time|what date|status of|is (the |my )?([\w.-]+ ){1,3}(up|down|running|online|"
                   r"healthy)|disk (space|usage)|uptime|current (version|price|weather|time)|weather|"
                   r"latest (version|release|commit|price)|exchange rate|word count|line count|"
                   r"list of|look up|lookup)\b")),
    (T_ORCHESTRATION, _rx(r"\b(for each|each of the|one by one|in parallel|and report (back|the results?|"
                          r"what you find)|return (the|a) (list|table|results?|urls?|paths?|raw))\b")),
)

# Politeness and addressing that precede the real leading verb.
_LEAD_FILLER_RX = re.compile(
    r"^(?:(?:hey|hi|hello|yo|ok|okay|so|um|please|pls|plz|kindly|quickly|quick|just|now|then|also|"
    r"can you|could you|would you|will you|can u|could u|i need you to|i want you to|i'd like you to|"
    r"go and|go|@\w+|\w+,)\s+)+",
    re.IGNORECASE,
)

# Scaffolding the runtime wraps around a prompt. Not the task; never classify it.
_SCAFFOLD_RX = re.compile(
    r"\[(?:IMPORTANT|CURRENT RUN TIME|SYSTEM|NOTE)[^\]]*\]\s*", re.IGNORECASE,
)
# cron/scheduler_prompt.py::_prepend_context_block: "## heading\nintro\n\n```\nbody\n```\n\n".
# The fenced body is collected data (script output, an upstream job's result), not the task.
_DATA_BLOCK_RX = re.compile(r"^## [^\n]+\n[^\n]*\n\n```\n.*?\n```\n*", re.MULTILINE | re.DOTALL)
# cron/notepad.py::render_notepad_section: the job's scratchpad and its CLI help text.
_NOTEPAD_RX = re.compile(
    r"^## Job notepad \(persistent across runs\)\n.*?\n\n(?:- [^\n]*\n)*\n?", re.MULTILINE | re.DOTALL,
)
_SENDER_PREFIX_RX = re.compile(r"^\[[^\]\n]{1,64}\]\s+")
# The cron delivery contract ("respond with [SILENT] if nothing is new") is plumbing.
_SILENT_RX = re.compile(r"\b(?:respond|reply|answer|say|return|output)\s+(?:with\s+)?(?:exactly\s+|only\s+)?"
                        r"[\"'`]*\[?silent\]?[\"'`]*", re.IGNORECASE)


@dataclass(frozen=True)
class TaskSignals:
    """Everything a routing decision may read. All free to compute locally."""

    instruction: str = ""            # the ask itself (turn message, or a child's goal)
    context: str = ""                # extra material (a child's context); premium signals only
    scope: str = SCOPE_TURN
    platform: str = ""
    is_background: bool = False      # cron / scheduled: no human waiting
    is_public_facing: bool = False   # a caller already knows the output is public
    is_protected: bool = False       # content a plugin marked protected
    has_attachments: bool = False    # images / audio / files on the turn
    toolsets: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    task_type: str
    reason: str
    # True when the heuristic itself is sure (a premium signal fired, or cheap evidence
    # with no premium signal). False = nothing decisive: premium by default, and the
    # only case an LLM router may be asked about.
    confident: bool

    @property
    def is_cheap(self) -> bool:
        return self.tier == TIER_CHEAP


def clean_instruction(text: str) -> str:
    """Strip runtime scaffolding (cron hint, run-time line, sender prefix, data blocks)."""
    if not isinstance(text, str):
        text = str(text or "")
    text = _SENDER_PREFIX_RX.sub("", text.lstrip(), count=1)
    text = _SCAFFOLD_RX.sub("", text)
    text = _DATA_BLOCK_RX.sub("", text)
    text = _NOTEPAD_RX.sub("", text)
    text = _SILENT_RX.sub("", text)
    return text.strip()


def _lead_verb(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    first_line = _LEAD_FILLER_RX.sub("", first_line.strip())
    words = first_line.split(maxsplit=1)
    return words[0].strip(".,!?:;\"'()*`").lower() if words else ""


def _premium_signal(text: str) -> Optional[str]:
    for task_type, rx in _PREMIUM_SIGNALS:
        if rx.search(text):
            return task_type
    return None


# Verbs that make a step more than a lookup. A cheap-looking ask with one of these in a
# LATER clause ("check the code and clean it up", "read the proposals and rank them")
# is a multi-step task whose hard step is the second one.
_ACTION_VERBS = frozenset({
    "answer", "approve", "ask", "book", "build", "call", "change", "choose", "clean", "comment",
    "compare", "contact", "create", "critique", "decide", "delete", "deploy", "draft", "edit",
    "email", "explain", "figure", "fix", "flag", "handle", "improve", "judge", "make", "merge",
    "notify", "order", "pick", "plan", "post", "prioritise", "prioritize", "propose", "rank",
    "rate", "refactor", "remove", "reply", "respond", "restart", "review", "rewrite", "schedule",
    "send", "solve", "sort", "suggest", "tidy", "translate", "update", "wake", "work", "write",
    "advise", "assess", "evaluate", "recommend", "resolve", "repair", "tweak", "adjust", "message",
    "dm", "publish", "share", "forward", "warn", "alert", "escalate", "buy", "pay",
    "synthesise", "synthesize", "conclude", "interpret", "infer", "judge", "argue", "weigh",
})
# "sort" is a cheap lead verb ("sort these by date") but a hard later step ("... and sort it
# out"). Only the later-clause position reads it as an action.
_CLAUSE_SPLIT_RX = re.compile(r"\b(?:and then|and|then|after that|also|but)\b|[;,.\n]", re.IGNORECASE)
_PRONOUN_LEAD_RX = re.compile(r"^(?:i|we|you|please|maybe|if|so|to)\s+", re.IGNORECASE)


def _later_step_is_hard(instruction: str) -> Optional[str]:
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RX.split(instruction) if c and c.strip()]
    for clause in clauses[1:]:
        clause = _PRONOUN_LEAD_RX.sub("", clause)
        verb = _lead_verb(clause)
        if verb in _ACTION_VERBS:
            return verb
    return None


def _cheap_evidence(instruction: str) -> Optional[str]:
    verb = _lead_verb(instruction)
    if verb in _CHEAP_LEAD_VERBS:
        return _CHEAP_LEAD_VERBS[verb]
    for task_type, rx in _CHEAP_PHRASES:
        if rx.search(instruction):
            return task_type
    return None


def _premium(task_type: str, why: str, confident: bool = True) -> RouteDecision:
    return RouteDecision(TIER_PREMIUM, task_type, why, confident)


def classify_task(sig: TaskSignals) -> RouteDecision:
    """Route one task. Premium unless there is cheap evidence and no premium signal."""
    if sig.is_protected:
        return _premium(T_PROTECTED, "protected content")
    if sig.is_public_facing:
        return _premium(T_PUBLIC_WRITING, "public-facing")
    if sig.has_attachments:
        return _premium(T_MULTIMODAL, "attachments")

    instruction = clean_instruction(sig.instruction)
    context = sig.context if isinstance(sig.context, str) else str(sig.context or "")
    if not instruction:
        return _premium(T_UNCERTAIN, "empty", confident=False)
    if len(instruction) + len(context) > _MAX_JUDGEABLE_CHARS:
        return _premium(T_UNCERTAIN, "too long to judge", confident=False)

    # Context-dependent follow-ups first: their meaning lives in earlier turns.
    if sig.scope == SCOPE_TURN and _CONTEXTUAL_RX.search(instruction):
        return _premium(T_CONTEXTUAL, "depends on earlier context")

    signal = _premium_signal(instruction + "\n" + context)
    if signal is not None:
        return _premium(signal, f"{signal} signal")

    evidence = _cheap_evidence(instruction)
    if evidence is None:
        return _premium(T_UNCERTAIN, "no cheap evidence", confident=False)

    hard_step = _later_step_is_hard(instruction)
    if hard_step is not None:
        return _premium(T_UNCERTAIN, "a later step needs more than a lookup")

    return RouteDecision(TIER_CHEAP, evidence, f"{evidence} evidence", True)
