# Contributing

Thanks for helping. This page covers the mechanics; [AGENTS.md](AGENTS.md) has the coding
conventions (tabs, `T*`/`I*` prefixes, full PHPDoc, PRADO exception classes with codes in
`config/errorMessages.txt`).

## Setup

```sh
git clone https://github.com/belisoful/prado-bayesian
cd prado-bayesian
composer install
```

PHP 8.1 to 8.5 with `ext-mbstring` (CI runs every version; keep 8.4 and 8.5 deprecation-free). `ext-pdo` with the SQLite driver runs the SQL suite
locally; `ext-redis` and the server-backed drivers are optional (below).

## The full check

Every change must pass the full check before it is proposed. It runs, in order, `php -l` over
`src/` and `tests/`, php-cs-fixer in dry-run mode, PHPStan at the level in `phpstan.neon.dist`,
and the unit suite:

```sh
composer fulltest
```

The individual steps are `composer lint`, `composer cs`, `composer stan` and `composer unittest`;
`composer fix` applies the code style. CI runs exactly these scripts, so a green `fulltest` is a
green build.

Bug fixes come with a failing test first. Tests live under `tests/unit`, one class per source
class, and cover the typical, edge and failure paths of what they test. Do not change the
phpunit options.

## Backend suites

The SQL and Redis tests skip when their backend is missing. Point them at real servers with
environment variables; the values below are the ones CI uses:

```sh
export BAYESIAN_MYSQL_DSN="mysql:host=127.0.0.1;port=3306;dbname=bayesian_test"
export BAYESIAN_MYSQL_USER=root
export BAYESIAN_MYSQL_PASSWORD=bayesian
export BAYESIAN_PGSQL_DSN="pgsql:host=127.0.0.1;port=5432;dbname=bayesian_test"
export BAYESIAN_PGSQL_USER=postgres
export BAYESIAN_PGSQL_PASSWORD=bayesian
export BAYESIAN_REQUIRE_BACKENDS=1   # turn every skip into a failure
composer unittest
```

Redis is found at `127.0.0.1:6379` when `ext-redis` is loaded. The test tables and keys carry
unique names and are dropped afterwards, but use a throwaway database and Redis instance all
the same. `BAYESIAN_REQUIRE_BACKENDS=1` is what CI sets, so a build cannot go green by skipping.

Coverage: `composer coverage` (needs Xdebug or PCOV) writes `build/logs/clover.xml`; CI enforces
the floor in `.github/workflows/prado-bayesian.yml` with `tests/test_tools/check-coverage.php`.

The Composer-extension integration check, `composer integration`, needs a PRADO checkout at
`../prado.master` (or a path as its first argument) and builds a throwaway consumer project.

## Benchmarks

`composer benchmark` reproduces the model-size and load-time figures quoted in
`docs/storage.md` on your machine. Quote figures in documentation only from that script, and say
which machine produced them.

## Changes to stored formats

A saved payload carries `formatVersion`, and a per-token model's metadata carries the storage's
`layoutVersion`. Any change to what is stored bumps the relevant version, keeps the loader
reading the previous version (upgrading in place where that is possible), and gets a migration
note in the CHANGELOG.

## Changelog and versions

Every behavior change goes under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md). The package is
pre-1.0: a minor release may change public APIs, and when it does the entry says what changed
and how to migrate; a patch release does not. New public symbols carry `@since` with the next
release version.
