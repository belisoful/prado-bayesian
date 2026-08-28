<?php

use Belisoful\Prado\Util\Bayesian\Tokenizer\TRegexTokenizer;

class TRegexTokenizerTest extends PHPUnit\Framework\TestCase
{
	public function testExtractsEmails()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/');
		$tokens = $tokenizer->tokenize('Email me at test@example.com or admin@foo.org please');
		self::assertSame(['test@example.com', 'admin@foo.org'], $tokens);
	}

	public function testLowercasingFlag()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/([a-zA-Z]+)/');
		$tokenizer->setLowercase(true);
		$tokens = $tokenizer->tokenize('Hello WORLD');
		self::assertSame(['hello', 'world'], $tokens);
	}

	public function testNoLowercasing()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/([a-zA-Z]+)/');
		$tokenizer->setLowercase(false);
		$tokens = $tokenizer->tokenize('Hello WORLD');
		self::assertSame(['Hello', 'WORLD'], $tokens);
	}

	public function testNoMatchesProducesEmpty()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/(\d{10})/');
		self::assertSame([], $tokenizer->tokenize('no numbers here'));
	}

	public function testDefaultPatternExtractsWords()
	{
		$tokenizer = new TRegexTokenizer();
		$tokens = $tokenizer->tokenize('Hello, world!');
		self::assertSame(['hello', 'world'], $tokens);
	}

	public function testPatternWithoutCapturingGroupUsesWholeMatch()
	{
		// A valid pattern with no parentheses must not silently produce an empty list.
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/\w+/');
		self::assertSame(['hello', 'world'], $tokenizer->tokenize('Hello world'));
	}

	public function testUnmatchedOptionalGroupYieldsNothing()
	{
		// 'foo' matches with group 1 unmatched (''), 'foobar' matches with group 1 = 'bar'.
		// The empty group must be dropped, not replaced with the whole match.
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/(?:foo)(bar)?/');
		self::assertSame(['bar'], $tokenizer->tokenize('foo foobar'));
	}

	public function testCapturingGroupYieldsOnlyTheGroup()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/#(\w+)/');
		self::assertSame(['php', 'prado'], $tokenizer->tokenize('Tags: #PHP #Prado plain'));
	}

	public function testSetPatternRejectsInvalidPattern()
	{
		$tokenizer = new TRegexTokenizer();
		$default = $tokenizer->getPattern();
		try {
			$tokenizer->setPattern('/(a+/');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
		self::assertSame($default, $tokenizer->getPattern());
	}

	public function testSetPatternRejectsEmptyPattern()
	{
		$tokenizer = new TRegexTokenizer();
		try {
			$tokenizer->setPattern('');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
	}

	public function testCatastrophicPatternThrowsPatternFailed()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/(a+)+$/');
		try {
			$tokenizer->tokenize(str_repeat('a', 5000) . 'b');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_failed', $e->getErrorCode());
		}
	}

	public function testInvalidUtf8InputDoesNotThrow()
	{
		$tokenizer = new TRegexTokenizer();
		$tokens = $tokenizer->tokenize("caf\xE9 ok");
		self::assertContains('ok', $tokens);
		foreach ($tokens as $token) {
			self::assertTrue(mb_check_encoding($token, 'UTF-8'));
		}
	}

	public function testExportConfigReflectsSettings()
	{
		$tokenizer = new TRegexTokenizer();
		self::assertSame(['pattern' => '/([\p{L}\p{N}]+)/u', 'lowercase' => true], $tokenizer->exportConfig());
		$tokenizer->setPattern('/(\d+)/');
		$tokenizer->setLowercase(false);
		self::assertSame(['pattern' => '/(\d+)/', 'lowercase' => false], $tokenizer->exportConfig());
	}

	public function testExportImportConfigRoundTrip()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->setPattern('/([A-Z]+)/');
		$tokenizer->setLowercase(false);
		$restored = new TRegexTokenizer();
		$restored->importConfig($tokenizer->exportConfig());
		self::assertSame('/([A-Z]+)/', $restored->getPattern());
		self::assertFalse($restored->getLowercase());
		self::assertSame($tokenizer->tokenize('ABC def GHI'), $restored->tokenize('ABC def GHI'));
		self::assertSame(['ABC', 'GHI'], $restored->tokenize('ABC def GHI'));
	}

	public function testImportConfigRejectsInvalidStoredPattern()
	{
		$tokenizer = new TRegexTokenizer();
		$default = $tokenizer->getPattern();
		try {
			$tokenizer->importConfig(['pattern' => 42]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
		self::assertSame($default, $tokenizer->getPattern());
	}

	public function testImportConfigCoercesScalarTypes()
	{
		$tokenizer = new TRegexTokenizer();
		$tokenizer->importConfig(['lowercase' => 0]);
		self::assertFalse($tokenizer->getLowercase());
		$tokenizer->importConfig(['lowercase' => '1', 'unknownKey' => 'ignored']);
		self::assertTrue($tokenizer->getLowercase());
	}

	public function testImportConfigRejectsInvalidPattern()
	{
		$tokenizer = new TRegexTokenizer();
		try {
			$tokenizer->importConfig(['pattern' => '/(a+/']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
	}
}
