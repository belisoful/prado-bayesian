# PRADO Bayesian Extension — Documentation

Reference and background material for `belisoful/prado-bayesian`. The top-level
[README](../README.md) is the quick start; these pages go deeper.

| Page | What it covers |
| --- | --- |
| [Concepts](concepts.md) | How the extension works: the pipeline, the three Naive Bayes event models, smoothing, TF-IDF, log-space arithmetic, and evaluation |
| [Class reference](classes.md) | Every public class and interface, grouped by namespace, with its role and public API |
| [Storage backends](storage.md) | The `IBayesianStorage` contract, the four backends, how to choose between them, and what a model costs in bytes and memory |
| [Configuration](configuration.md) | Wiring the module and the HTTP service into a PRADO application (XML and PHP forms), and the full error-code list |

## Where things live

All classes are namespaced under `Prado\Util\Bayesian`, except the HTTP service, which lives
at `Prado\Web\Services\TBayesianService` where PRADO expects services to be.

```
src/Util/Bayesian/
├── Classifier/     IBayesianClassifier and the four classifier classes
├── Tokenizer/      IBayesianTokenizer, the tokenizers, the chain, the factory, the shared trait
├── Math/           TBayesMath (log-space arithmetic), TFIdf (term weighting)
├── Evaluation/     TConfusionMatrix, TBayesianMetrics
├── Storage/        IBayesianStorage and the four storage backends
└── *.php           TBayesianModule, TBayesianRecommender, and the training/vocabulary types
```

Every class is registered in `config/prado-bayesian-classes.json` under its short name, so
application configuration can say `class="TNaiveBayesClassifier"` instead of spelling out the
fully-qualified name. See [Configuration](configuration.md#short-class-names).

## API documentation

Every public method carries a complete PHPDoc block — description, `@param`, `@return`, and
`@throws` — so the source doubles as the API reference. `phpdocumentor/shim` is a dev
dependency; it installs the phpDocumentor PHAR into `vendor/bin/phpdoc` on `composer install`
(the PHAR is downloaded separately from the Composer package, so it is absent in checkouts
where that step did not run). With it present:

```bash
vendor/bin/phpdoc -d src -t docs/api
```

`docs/api/` is generated output: php-cs-fixer already excludes `docs/`, and `.gitignore`
excludes `/docs/api` so the rendered HTML is not committed.
