# PRADO Bayesian Extension Agent Guidelines

## Build, Lint, and Test Commands

### Running Tests
- **All Unit Tests**: `vendor/bin/phpunit --testsuite unit` - runs all unit tests
- **Test Filter**: `vendor/bin/phpunit --testsuite unit --filter <test function, class, or directory>`

### Linting and Code Analysis
- **PHPStan Analysis**: `vendor/bin/phpstan analyse src/ --memory-limit=512M`
- **PHP CS Fixer (Dry-run)**: `vendor/bin/php-cs-fixer fix --dry-run src/` (check)
- **PHP CS Fixer (Fix)**: `vendor/bin/php-cs-fixer fix src/` (apply fixes)

### Build Commands
- **Install Dependencies**: `composer install` - installs all dependencies
- **Updating Dependencies**: `composer update` - updates dependencies

## Code Style Guidelines
- "if" has a statement block after
- Use php-cs-fixer to correct code styles

### PHP Coding Standards
- Follow PSR-4 autoloading standard
- All PHP files must begin with `<?php` tag (short open tags not allowed)
- Use 1 tab for indentations (no spaces)
- All class names must be in PascalCase
- All method names must be in camelCase
- All variable names must be in camelCase
- Constants must be in SCREAMING_SNAKE_CASE
- All class properties must be declared with visibility modifiers (public, protected, private)

### Naming Conventions
- Class names: `TPascalCase` (e.g., `TComponent`)
- Class name prefix: `T*` (e.g., `TApplication`)
- Method names: `camelCase` (e.g., `getComponent`)
- Variables: `camelCase` (e.g., `$componentName`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_RETRY_COUNT`)
- Namespace: `Prado\{Module}` (e.g., `Prado\Web\UI\TControl`)
- Interface prefix: `I*` (e.g., `IBayesianClassifier`)
- Template file extension: ".tpl"
- Web Page template file extension: ".page"

### Documentation Standards
- All public methods must have PHPDoc comments with:
  - `@param` for parameters
  - `@return` for return values
  - `@throws` for exceptions
- Classes must have a clear and comprehensive docblock at the top with class description with:
  - Examples, where necessary
  - `@author` for attribution
  - `@since` for version
  - `@method` for dynamic events with prefix 'dy-'; which are called (on "$this->dy-") but not defined.
- Inline comments should be in English and start with `//`
- When documenting new methods or classes with "@since" use the next release version.
- All documentation should be written in present perfect tense

### Error Handling
- Use try/catch blocks for operations that can fail
- Throw appropriate PRADO exceptions (`TInvalidDataValueException`, `TInvalidOperationException`, `TConfigurationException`, etc.)
- Return false or null for methods that are designed to fail gracefully
- All methods should handle edge cases and validate input parameters
- Extension Exceptions use errorCodes specified in `config/errorMessages.txt`; messages.txt is purely for user information display only.

### Imports and Includes
- Use PSR-4 autoloading - no manual includes required
- All framework classes are accessed via namespace prefixes
- Third-party libraries are loaded via Composer
- Use proper `use` statements for namespaces at the top of PHP files

### Framework Specific Guidelines
- All components inherit from `TComponent` base class
- `TComponent` has features for dynamic event and extension by attached Behaviors (__call, __callStatic), dynamic properties (__get, __set, __isset, __unset), __clone, __sleep, __wakeup, and _getZappableSleepProps
- Behaviors can be attached to any `TComponent` to alter its behavior and functionality.
- Use the event-driven programming model with events; like `onLoad`, `onInit`, `onPreRender`
- Methods with prefix 'dy' are dynamic events to call attached and active Behaviors; like 'dyShouldContinue', 'dyClone', and 'dyValidate'
- Called Dynamic Events must be documented in the class phpdoc with "@method"
- Dynamic event are implemented by attached behaviors not in the calling class
- The first parameter of a dynamic event is always filtered and returned.
- Optional class methods can directly be called on non-behavior classes as "dynamic events"
- Methods with prefix 'fx' are global events that may or may not be automatically registered depending on getAutoGlobalListen(); like 'fxAttachClassBehavior'
- getAutoGlobalListen() is optimized by class hierarchy for utility and performance
- All events are raised in specified priority order
- XML and PHP is supported for application configuration
- 'framework/classes.php' MUST be updated with all new classes.
- A full check consists of the 4 checks (in order): `php -l` compile, php-cs-fixer, phpstan, phpunit (all checks must pass successfully)
- A full check must be done for code to be ready for git commit.
- The current version is 0.1.0 (pre-release). New classes/methods use `@since 0.1.0`; the first stable release will be 1.0.0.
- This extension namespaces its classes under `Belisoful\Prado\Util\Bayesian\` (PSR-4 `Belisoful\Prado\` → `src/`); the `Prado\` prefix belongs to the framework and is never written to by this package. Extensions do NOT update the framework's `classes.php`.
- Error codes live in `config/errorMessages.txt`, registered system-wide via `extra.prado.error-messages` in `composer.json`; the framework's `messages.txt` is not used.
- The Prado3-style short class name → PHP FQN class map lives in `config/prado-bayesian-classes.json`, registered system-wide via `extra.prado.class-map` in `composer.json`.

## Testing Guidelines
- The testing platform is "phpunit"
- All new code must include unit tests
- Unit test functions must comprehensively assert both typical and edge cases
- Maximal coverage of code execution paths of a class is required
- Test error conditions and exception handling
- Use mock objects where appropriate
- Functional tests should verify complete user workflows
- Tests should be isolated from each other (no shared state)
- When unit testing one or cluster of classes, only run the unit tests for that class or cluster/directory.
- NEVER add/change phpunit command options when unit testing; only run project unit tests as specified

## Development Environment
- PHP 8.1 or higher required
- PHP extensions: ctype, dom, intl, json, mbstring, pcre, spl (typical)
- Optional extensions for additional features: pdo, redis
- Composer for dependency management
- Required developer dependencies for code checking: phpunit/phpunit, phpstan/phpstan, friendsofphp/php-cs-fixer
- Presume that project dependencies are installed
- `pradosoft/prado` is a runtime requirement at `^4.4@dev`, resolved from the framework's GitHub repository until 4.4 reaches Packagist; the constraint matches `dev-master` through the branch alias PRADO declares (`dev-master` → `4.4.x-dev`), which is why `minimum-stability` is `dev` and `prefer-stable` is `true`. No machine-specific path is committed. To work against a local checkout, add a path repository to your working copy without committing it

## Cursor/Copilot Instructions
No specific Cursor or Copilot rules currently defined for this project.

# PRADO Framework Agent Safeguards -- ANTI-PATTERNS
Between the next brackets, it is required without exception:
{
- NEVER (without exception) execute the following "git" commands without asking the developer for approval first: clone, checkout, mv, restore, rm, branch, add, commit, merge, rebase, reset, pull, push, fetch
- NEVER (without exception) execute "rm" commands on any paths without asking the developer for approval first
- NEVER remove composer --dev dependencies because those are a required for development on the Project
- NEVER perform an action that erases or overwrites files for the task of unit testing and fixing; file changes are important and must be kept, because the changes themselves are being unit tested.
- NEVER delete any folders or files until the associated task is absolutely and totally complete.
}
