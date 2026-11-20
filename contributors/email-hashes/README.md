# Hashed contributor allowances (fork-only)

`scripts/ci/check_contributor_emails.py` fails a PR whose commits carry an author email with
no mapping. Upstream maps an email by publishing it as a file name under
`contributors/emails/`. This fork does not publish its contributors' addresses, so a known
author is allowed here instead:

- file name: the sha256 hex digest of the lowercased commit email
- content: the author's GitHub login

Prefer committing with your GitHub noreply address (`<id>+<login>@users.noreply.github.com`),
which the check accepts with no entry at all. Add a hash only for an address that commits
already carry.

```
python3 -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].strip().casefold().encode()).hexdigest())" <email>
```

This directory does not exist upstream, so upstream syncs never touch it. Release-note
attribution (`scripts/releases/authors.py`) does not read it and falls back to the git name.
