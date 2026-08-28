<?php

use Belisoful\Prado\Util\Bayesian\Tokenizer\IBayesianTokenizer;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerChain;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerFactory;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TRegexTokenizer;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TWordTokenizer;
use Prado\Exceptions\TInvalidDataValueException;

class TBayesianTokenizerFactoryTest extends PHPUnit\Framework\TestCase
{
	public function testExportRecordsClassAndConfig()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setMinLength(3);
		$tokenizer->setStopWords(['the', 'and']);
		$state = TBayesianTokenizerFactory::export($tokenizer);
		self::assertSame(TWordTokenizer::class, $state['class']);
		self::assertSame($tokenizer->exportConfig(), $state['config']);
		self::assertSame(3, $state['config']['minLength']);
		self::assertSame(['the', 'and'], $state['config']['stopWords']);
	}

	public function testRestoreCreatesConfiguredInstance()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(5);
		$tokenizer->setCharacters(false);
		$tokenizer->setPad(false);
		$state = TBayesianTokenizerFactory::export($tokenizer);
		$restored = TBayesianTokenizerFactory::restore($state);
		self::assertInstanceOf(TNGramTokenizer::class, $restored);
		self::assertNotSame($tokenizer, $restored);
		self::assertSame(5, $restored->getN());
		self::assertFalse($restored->getCharacters());
		self::assertFalse($restored->getPad());
	}

	public function testRestoreAcceptsLeadingBackslashInClassName()
	{
		$restored = TBayesianTokenizerFactory::restore(['class' => '\\' . TRegexTokenizer::class, 'config' => ['lowercase' => false]]);
		self::assertInstanceOf(TRegexTokenizer::class, $restored);
		self::assertFalse($restored->getLowercase());
	}

	public function testRestoreReturnsNullWithoutClass()
	{
		self::assertNull(TBayesianTokenizerFactory::restore([]));
		self::assertNull(TBayesianTokenizerFactory::restore(['class' => '', 'config' => ['minLength' => 3]]));
		self::assertNull(TBayesianTokenizerFactory::restore(['class' => null]));
		self::assertNull(TBayesianTokenizerFactory::restore(['class' => 42]));
	}

	public function testRestoreReturnsNullWithoutClassEvenWhenCurrentGiven()
	{
		$current = new TWordTokenizer();
		self::assertNull(TBayesianTokenizerFactory::restore(['config' => ['minLength' => 9]], $current));
		self::assertSame(2, $current->getMinLength(), 'the current tokenizer is left untouched');
	}

	public function testRestoreThrowsForMissingClass()
	{
		try {
			TBayesianTokenizerFactory::restore(['class' => 'Nope\\Missing\\Tokenizer']);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_class_invalid', $e->getErrorCode());
		}
	}

	public function testRestoreThrowsForNonTokenizerClass()
	{
		try {
			TBayesianTokenizerFactory::restore(['class' => \stdClass::class]);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_class_invalid', $e->getErrorCode());
		}
	}

	public function testRestoreReusesCurrentWhenClassMatches()
	{
		$current = new TWordTokenizer();
		$restored = TBayesianTokenizerFactory::restore(
			['class' => TWordTokenizer::class, 'config' => ['minLength' => 4, 'stopWords' => ['foo']]],
			$current
		);
		self::assertSame($current, $restored);
		self::assertSame(4, $current->getMinLength());
		self::assertSame(['foo'], $current->getStopWords());
	}

	public function testRestoreReplacesCurrentWhenClassDiffers()
	{
		$current = new TWordTokenizer();
		$restored = TBayesianTokenizerFactory::restore(['class' => TRegexTokenizer::class, 'config' => []], $current);
		self::assertInstanceOf(TRegexTokenizer::class, $restored);
		self::assertNotSame($current, $restored);
	}

	public function testRestoreIgnoresNonArrayConfig()
	{
		$restored = TBayesianTokenizerFactory::restore(['class' => TWordTokenizer::class, 'config' => 'garbage']);
		self::assertInstanceOf(TWordTokenizer::class, $restored);
		self::assertSame(2, $restored->getMinLength());
	}

	public function testExportRestoreRoundTripForEveryTokenizer()
	{
		$chain = new TBayesianTokenizerChain();
		$chain->addTokenizer(new TWordTokenizer());
		$chain->addTokenizer(new TNGramTokenizer());
		foreach ([new TWordTokenizer(), new TRegexTokenizer(), new TNGramTokenizer(), $chain] as $tokenizer) {
			$restored = TBayesianTokenizerFactory::restore(TBayesianTokenizerFactory::export($tokenizer));
			self::assertInstanceOf(IBayesianTokenizer::class, $restored);
			self::assertSame($tokenizer::class, $restored::class);
			self::assertSame($tokenizer->exportConfig(), $restored->exportConfig());
			self::assertSame($tokenizer->tokenize('Hello wonderful world'), $restored->tokenize('Hello wonderful world'));
		}
	}

	public function testScrubTextReplacesInvalidUtf8()
	{
		$input = "caf\xE9 ok";
		self::assertFalse(mb_check_encoding($input, 'UTF-8'));
		$scrubbed = TBayesianTokenizerFactory::scrubText($input);
		self::assertTrue(mb_check_encoding($scrubbed, 'UTF-8'));
		self::assertStringContainsString('caf', $scrubbed);
		self::assertStringContainsString(' ok', $scrubbed);
	}

	public function testScrubTextReturnsValidTextUnchanged()
	{
		self::assertSame('', TBayesianTokenizerFactory::scrubText(''));
		self::assertSame('café ok 啊', TBayesianTokenizerFactory::scrubText('café ok 啊'));
	}

	public function testAssertPatternAcceptsValidPattern()
	{
		TBayesianTokenizerFactory::assertPattern('/[a-z]+/u');
		TBayesianTokenizerFactory::assertPattern('/(a+)+$/');
		$this->addToAssertionCount(1);
	}

	public function testAssertPatternRejectsUnbalancedPattern()
	{
		try {
			TBayesianTokenizerFactory::assertPattern('/(a+/');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
			self::assertStringContainsString('/(a+/', $e->getErrorMessage());
		}
	}

	public function testAssertPatternRejectsEmptyPattern()
	{
		try {
			TBayesianTokenizerFactory::assertPattern('');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
	}

	public function testAssertPatternRejectsMissingDelimiter()
	{
		try {
			TBayesianTokenizerFactory::assertPattern('abc');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
	}

	public function testAssertPatternDoesNotLeakErrorHandler()
	{
		$handler = static fn (): bool => true;
		set_error_handler($handler);
		try {
			try {
				TBayesianTokenizerFactory::assertPattern('/(a+/');
			} catch (TInvalidDataValueException $e) {
				// expected
			}
			// restore_error_handler() inside the factory must return to OUR handler, not remove it.
			$active = set_error_handler(null);
			restore_error_handler();
			self::assertSame($handler, $active);
		} finally {
			restore_error_handler();
		}
	}

	public function testCheckPregErrorIsSilentAfterSuccessfulMatch()
	{
		preg_match('/a/', 'a');
		TBayesianTokenizerFactory::checkPregError('/a/');
		$this->addToAssertionCount(1);
	}

	public function testCheckPregErrorThrowsAfterBacktrackLimit()
	{
		$result = @preg_match('/(a+)+$/', str_repeat('a', 5000) . 'b');
		self::assertFalse($result);
		try {
			TBayesianTokenizerFactory::checkPregError('/(a+)+$/');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_failed', $e->getErrorCode());
			self::assertStringContainsString('/(a+)+$/', $e->getErrorMessage());
		}
	}
}
