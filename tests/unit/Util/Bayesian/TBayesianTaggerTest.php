<?php

use Belisoful\Prado\Util\Bayesian\Calibration\TPlattScaling;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TMemoryBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianTagger;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

class TBayesianTaggerTest extends PHPUnit\Framework\TestCase
{
	/** @var string[] Files to remove after the test. */
	private array $_files = [];

	protected function tearDown(): void
	{
		foreach ($this->_files as $file) {
			@unlink($file);
		}
		$this->_files = [];
	}

	/** @return array<int, array{0: string[], 1: string}> A small tagged corpus. */
	private function corpus(): array
	{
		return [
			[['php', 'security'], 'validate every request parameter before it reaches the sql query'],
			[['php'], 'composer autoloads the classes of the namespace'],
			[['security'], 'rotate the password and revoke the leaked token'],
			[['cooking'], 'simmer the tomato sauce for twenty minutes'],
			[['cooking'], 'whisk the eggs and fold in the flour'],
			[[], 'the meeting moved to tuesday afternoon'],
		];
	}

	private function trainedTagger(): TBayesianTagger
	{
		$tagger = new TBayesianTagger();
		foreach ($this->corpus() as [$labels, $document]) {
			$tagger->train($labels, $document);
		}
		return $tagger;
	}

	public function testDefaultsAndProperties()
	{
		$tagger = new TBayesianTagger();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $tagger->getClassifier());
		self::assertSame('*', $tagger->getBackgroundCategory());
		self::assertSame(0.5, $tagger->getThreshold());
		self::assertSame(0, $tagger->getMaxTags());
		self::assertFalse($tagger->getIsTrained());
		self::assertSame([], $tagger->getLabels());
		self::assertFalse($tagger->getIsCalibrated());
		$tagger->setThreshold(1.5);
		self::assertSame(1.0, $tagger->getThreshold());
		$tagger->setThreshold(-1.0);
		self::assertSame(0.0, $tagger->getThreshold());
		$tagger->setThreshold(NAN);
		self::assertSame(0.5, $tagger->getThreshold());
		$tagger->setMaxTags(-2);
		self::assertSame(0, $tagger->getMaxTags());
		$tagger->setBackgroundCategory('__all');
		self::assertSame('__all', $tagger->getBackgroundCategory());
		$classifier = new TNaiveBayesClassifier();
		$tagger->setClassifier($classifier);
		self::assertSame($classifier, $tagger->getClassifier());
		$this->expectException(TInvalidDataValueException::class);
		$tagger->setBackgroundCategory('');
	}

	public function testTrainingCountsEveryDocumentInTheBackgroundAndEachLabel()
	{
		$tagger = $this->trainedTagger();
		self::assertTrue($tagger->getIsTrained());
		self::assertSame(['php', 'security', 'cooking'], $tagger->getLabels());
		$vocabulary = $tagger->getClassifier()->getVocabulary();
		self::assertSame(6, $vocabulary->getCategory('*')->getDocumentCount(), 'every document, once');
		self::assertSame(2, $vocabulary->getCategory('php')->getDocumentCount());
		self::assertSame(2, $vocabulary->getCategory('security')->getDocumentCount());
		self::assertSame(1, $vocabulary->getCategory('*')->getTokenDocumentCount('sql'), 'the background document frequency counts a document once, whatever its labels');
	}

	public function testProbabilitiesAreIndependentPerLabel()
	{
		$tagger = $this->trainedTagger();
		$probabilities = $tagger->probabilities('escape the parameters of every sql query');
		self::assertSame(['php', 'security', 'cooking'], array_keys($probabilities));
		foreach ($probabilities as $label => $probability) {
			self::assertGreaterThanOrEqual(0.0, $probability, $label);
			self::assertLessThanOrEqual(1.0, $probability, $label);
		}
		self::assertGreaterThan($probabilities['cooking'], $probabilities['php']);
		self::assertGreaterThan($probabilities['cooking'], $probabilities['security']);
		self::assertGreaterThan(0.5, $probabilities['php']);
		self::assertGreaterThan(0.5, $probabilities['security'], 'two labels can both apply');
		self::assertLessThan(0.5, $probabilities['cooking']);
		$cooking = $tagger->probabilities('whisk the sauce and simmer the eggs');
		self::assertGreaterThan(0.5, $cooking['cooking']);
		self::assertLessThan(0.5, $cooking['php']);
		// The log-odds are what the probabilities come from.
		$logOdds = $tagger->logOdds('escape the parameters of every sql query');
		self::assertSame(array_keys($probabilities), array_keys($logOdds));
		self::assertEqualsWithDelta(1.0 / (1.0 + exp(-$logOdds['php'])), $probabilities['php'], 1e-12);
	}

	public function testOutOfVocabularyDocumentsFallBackToThePriors()
	{
		$tagger = $this->trainedTagger();
		$probabilities = $tagger->probabilities('entirely unknown wording here');
		// Two of six documents carry php: prior odds (2+1)/(4+1).
		self::assertEqualsWithDelta(1.0 / (1.0 + exp(-log(3.0 / 5.0))), $probabilities['php'], 1e-12);
		self::assertSame($probabilities, $tagger->probabilities(''));
		self::assertSame($probabilities, $tagger->probabilities(['unknown', 'tokens']));
	}

	public function testTagFiltersSortsAndLimits()
	{
		$tagger = $this->trainedTagger();
		$document = 'escape the parameters of every sql query';
		$tags = $tagger->tag($document);
		self::assertSame(['php', 'security'], array_keys($tags), 'above the threshold, highest first');
		$probabilities = $tagger->probabilities($document);
		self::assertSame($probabilities['php'], $tags['php']);
		self::assertGreaterThanOrEqual($tags['security'], $tags['php']);

		$tagger->setMaxTags(1);
		self::assertSame(['php'], array_keys($tagger->tag($document)));
		$tagger->setMaxTags(0);
		$tagger->setThreshold(0.0);
		self::assertSame(3, count($tagger->tag($document)), 'a zero threshold returns every label');
		$tagger->setThreshold(1.0);
		self::assertSame([], $tagger->tag($document));
	}

	public function testLabelsAreValidated()
	{
		$tagger = new TBayesianTagger();
		try {
			$tagger->train(['php', '*'], 'text');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tagger_label_reserved', $e->getErrorCode());
		}
		self::assertFalse($tagger->getIsTrained(), 'nothing was trained');
		try {
			$tagger->train(['php', ''], 'text');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_category_required', $e->getErrorCode());
		}
		// Repeated labels count once.
		$tagger->train(['php', 'php'], 'composer autoload');
		self::assertSame(1, $tagger->getClassifier()->getVocabulary()->getCategory('php')->getDocumentCount());
	}

	public function testUntrainedTaggerThrows()
	{
		$tagger = new TBayesianTagger();
		try {
			$tagger->probabilities('anything');
			self::fail('expected exception');
		} catch (TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
		$this->expectException(TInvalidOperationException::class);
		$tagger->tag('anything');
	}

	public function testUntrainWithdrawsADocumentExactly()
	{
		$reference = new TBayesianTagger();
		$full = new TBayesianTagger();
		foreach ($this->corpus() as $i => [$labels, $document]) {
			$full->train($labels, $document);
			if ($i !== 0) {
				$reference->train($labels, $document);
			}
		}
		$full->untrain(['php', 'security'], 'validate every request parameter before it reaches the sql query');
		self::assertSame($reference->getLabels(), $full->getLabels());
		foreach (['escape the sql query', 'simmer the sauce', 'nothing known'] as $probe) {
			self::assertSame($reference->probabilities($probe), $full->probabilities($probe), $probe);
		}
		// Untraining everything empties the tagger.
		foreach ($this->corpus() as $i => [$labels, $document]) {
			if ($i !== 0) {
				$full->untrain($labels, $document);
			}
		}
		self::assertFalse($full->getIsTrained());
		self::assertSame([], $full->getLabels());
	}

	public function testCalibrationFitsPerLabelAndPersistsWithTheModel()
	{
		$tagger = $this->trainedTagger();
		$storage = new TMemoryBayesianStorage();
		$tagger->getClassifier()->setStorage($storage);
		$tagger->getClassifier()->setName('tags');
		$heldOut = [
			[['php'], 'the sql query parameters of the request'],
			[['security'], 'revoke the token and rotate the password'],
			[['cooking'], 'fold the flour into the sauce'],
			[['php', 'security'], 'validate the request before the query'],
			[[], 'tuesday afternoon meeting'],
			[['cooking'], 'whisk twenty eggs'],
		];
		$raw = $tagger->probabilities('escape the sql query');
		$fitted = $tagger->calibrate($heldOut);
		self::assertSame(['php', 'security', 'cooking'], array_keys($fitted));
		foreach ($fitted as $scaling) {
			self::assertInstanceOf(TPlattScaling::class, $scaling);
			self::assertSame(6, $scaling->getSampleCount());
		}
		self::assertTrue($tagger->getIsCalibrated());
		$calibrated = $tagger->probabilities('escape the sql query');
		self::assertNotSame($raw, $calibrated);
		self::assertSame($fitted['php']->apply($tagger->logOdds('escape the sql query')['php']), $calibrated['php']);
		self::assertGreaterThan($calibrated['cooking'], $calibrated['php'], 'calibration keeps the order');

		$tagger->getClassifier()->save();
		$loaded = new TBayesianTagger();
		$loaded->getClassifier()->setStorage($storage);
		$loaded->getClassifier()->load('tags');
		self::assertTrue($loaded->getIsCalibrated());
		self::assertSame($calibrated, $loaded->probabilities('escape the sql query'));
		self::assertSame($tagger->tag('escape the sql query'), $loaded->tag('escape the sql query'));

		$loaded->setCalibrations(null);
		self::assertFalse($loaded->getIsCalibrated());
		self::assertSame($raw, $loaded->probabilities('escape the sql query'));
		$this->expectException(TInvalidDataValueException::class);
		$tagger->calibrate([]);
	}

	public function testPerTokenStorageTagsIdenticallyToThePayloadLayout()
	{
		BayesianBackends::requireBackend($this, extension_loaded('pdo') && in_array('sqlite', \PDO::getAvailableDrivers(), true), 'pdo_sqlite not available');
		$file = sys_get_temp_dir() . '/bayesian-tagger-' . uniqid('', true) . '.sqlite';
		$this->_files[] = $file;
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString('sqlite:' . $file);
		$storage->setMode(TSqlBayesianStorage::MODE_TOKEN);

		$resident = $this->trainedTagger();
		$resident->getClassifier()->setStorage($storage);
		$resident->getClassifier()->setName('tags');
		$resident->calibrate([[['php'], 'sql query parameters'], [['cooking'], 'simmer the sauce'], [[], 'tuesday meeting']]);
		$resident->getClassifier()->save();

		$lazy = new TBayesianTagger();
		$lazy->getClassifier()->setStorage($storage);
		$lazy->getClassifier()->load('tags');
		self::assertFalse($lazy->getClassifier()->getVocabulary()->getSupportsFullScan());
		self::assertSame($resident->getLabels(), $lazy->getLabels());
		self::assertTrue($lazy->getIsCalibrated());
		foreach (['escape the sql query', 'whisk the eggs', 'unknown words', ''] as $probe) {
			self::assertSame($resident->probabilities($probe), $lazy->probabilities($probe), $probe);
			self::assertSame($resident->tag($probe), $lazy->tag($probe), $probe);
		}
		// Incremental training and untraining through storage keep the two in step.
		$lazy->train(['cooking'], 'bake the bread until golden');
		$resident->train(['cooking'], 'bake the bread until golden');
		self::assertSame($resident->probabilities('bake the golden bread'), $lazy->probabilities('bake the golden bread'));
		$lazy->untrain(['cooking'], 'bake the bread until golden');
		$resident->untrain(['cooking'], 'bake the bread until golden');
		self::assertSame($resident->probabilities('bake the golden bread'), $lazy->probabilities('bake the golden bread'));
	}

	public function testTfidfWeightingIsHonoured()
	{
		$weighted = $this->trainedTagger();
		$plain = $this->trainedTagger();
		$plain->getClassifier()->setUseTfidf(false);
		self::assertNotSame($weighted->logOdds('sql sql sql query'), $plain->logOdds('sql sql sql query'));
		self::assertSame(array_keys($weighted->logOdds('x')), array_keys($plain->logOdds('x')));
	}
}
