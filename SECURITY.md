# Security policy

## Supported versions

`belisoful/prado-bayesian` is pre-1.0. Only the latest released minor version receives fixes;
there are no maintenance branches for earlier versions.

| Version | Supported |
|---|---|
| latest 0.x release | yes |
| earlier 0.x releases | no, upgrade |

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately to
**belisoful@icloud.com** with:

- the version (`composer show belisoful/prado-bayesian`), PHP version and storage backend;
- what an attacker can do, and the steps or a script that shows it;
- whether you want to be credited in the fix's release notes.

You will get an acknowledgement within a week. Confirmed issues are fixed in a patch release
with a CHANGELOG entry; you will be told when it ships and may then disclose.

## What the package does and does not protect

- **`TBayesianService` enforces no access control by default.** It answers every request that
  reaches it, each request costs a classification, and the scores describe the model. Restrict
  it with its authorization rules or PRADO's permissions manager before exposing it, or keep it
  behind a proxy that does; see [docs/configuration.md](docs/configuration.md#access-control).
  Its input is bounded by `MaxTextLength` and `MaxCandidates`.
- **Model names and SQL table names are validated** before they reach the filesystem or a SQL
  statement (no path separators, dots or null bytes in a file name; a plain identifier of at most
  48 characters for a table). Tokens and category names are only ever bound parameters.
- **A saved model contains the tokens of its training data.** Treat model files, rows and Redis
  keys with the same care as the text they were trained on: set `FileMode`/`DirectoryMode` on the
  file backend, and use a database user or Redis ACL that only this application holds.
- **Loading a model instantiates the tokenizer class it names**, but only a class that exists
  and implements `IBayesianTokenizer`; a payload cannot make the loader construct an arbitrary
  class. Do not load models from sources you do not trust anyway: a crafted payload can still
  choose the tokenizer settings and every count.
- **Untrusted input reaches PCRE** through the tokenizers. A pattern that hits the backtrack
  limit throws rather than returning an empty token list, and invalid UTF-8 is scrubbed first.
