<?php

use Belisoful\Prado\Util\Bayesian\Tokenizer\TWordTokenizer;

class TWordTokenizerTest extends PHPUnit\Framework\TestCase
{
	public function testLowercasesTokens()
	{
		$tokenizer = new TWordTokenizer();
		$tokens = $tokenizer->tokenize('Hello World');
		self::assertSame(['hello', 'world'], $tokens);
	}

	public function testSplitsOnPunctuation()
	{
		$tokenizer = new TWordTokenizer();
		$tokens = $tokenizer->tokenize('Hello, world! How are you?');
		self::assertSame(['hello', 'world', 'how', 'are', 'you'], $tokens);
	}

	public function testDropsShortTokens()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setMinLength(3);
		$tokens = $tokenizer->tokenize('a an the apple banana');
		self::assertSame(['the', 'apple', 'banana'], $tokens);
	}

	public function testDropsStopWords()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setStopWords(['the', 'and']);
		$tokens = $tokenizer->tokenize('the quick brown fox and the lazy dog');
		self::assertSame(['quick', 'brown', 'fox', 'lazy', 'dog'], $tokens);
	}

	public function testEmptyInputProducesEmptyOutput()
	{
		$tokenizer = new TWordTokenizer();
		self::assertSame([], $tokenizer->tokenize(''));
	}

	public function testOnlyPunctuationProducesEmptyOutput()
	{
		$tokenizer = new TWordTokenizer();
		self::assertSame([], $tokenizer->tokenize('!@#$%^&*()'));
	}

	public function testMultibyteTokens()
	{
		$tokenizer = new TWordTokenizer();
		$tokens = $tokenizer->tokenize('café résumé naïve');
		self::assertSame(['café', 'résumé', 'naïve'], $tokens);
	}

	public function testPreservesRepeatedTokens()
	{
		$tokenizer = new TWordTokenizer();
		$tokens = $tokenizer->tokenize('spam spam eggs spam');
		self::assertSame(['spam', 'spam', 'eggs', 'spam'], $tokens);
	}

	public function testStopWordsNullByDefault()
	{
		$tokenizer = new TWordTokenizer();
		self::assertNull($tokenizer->getStopWords());
	}

	public function testStopWordsLowercasedOnSet()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setStopWords(['THE', 'And']);
		$tokens = $tokenizer->tokenize('the cat and the dog');
		self::assertSame(['cat', 'dog'], $tokens);
	}

	public function testEmptyStopWordListDisablesFiltering()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setStopWords([]);
		$tokens = $tokenizer->tokenize('the cat');
		self::assertSame(['the', 'cat'], $tokens);
	}

	public function testCustomPattern()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setPattern('/\d+/');
		$tokens = $tokenizer->tokenize('order 12345 and item 678');
		self::assertSame(['12345', '678'], $tokens);
	}

	public function testMinLengthClampedToOne()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setMinLength(0);
		self::assertSame(1, $tokenizer->getMinLength());
	}

	public function testMinLengthUsesCharacterCountNotByteCount()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setMinLength(2);
		// '啊' is 1 character but 3 bytes in UTF-8; it should be dropped with minLength=2.
		$tokens = $tokenizer->tokenize('啊 foo');
		self::assertSame(['foo'], $tokens);
	}

	public function testMinLengthAllowsSingleMultibyteCharacter()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setMinLength(1);
		// With minLength=1, a single CJK character passes.
		$tokens = $tokenizer->tokenize('啊 foo');
		self::assertContains('啊', $tokens);
		self::assertContains('foo', $tokens);
	}

	public function testSetPatternRejectsInvalidPattern()
	{
		$tokenizer = new TWordTokenizer();
		$default = $tokenizer->getPattern();
		try {
			$tokenizer->setPattern('/(a+/');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
		self::assertSame($default, $tokenizer->getPattern(), 'a rejected pattern leaves the old one in place');
	}

	public function testSetPatternRejectsEmptyPattern()
	{
		$tokenizer = new TWordTokenizer();
		try {
			$tokenizer->setPattern('');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_pattern_invalid', $e->getErrorCode());
		}
	}

	public function testCatastrophicPatternThrowsPatternFailed()
	{
		$tokenizer = new TWordTokenizer();
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
		$tokenizer = new TWordTokenizer();
		$tokens = $tokenizer->tokenize("caf\xE9 ok");
		self::assertContains('ok', $tokens);
		foreach ($tokens as $token) {
			self::assertTrue(mb_check_encoding($token, 'UTF-8'));
		}
	}

	public function testExportConfigReflectsSettings()
	{
		$tokenizer = new TWordTokenizer();
		self::assertSame(['minLength' => 2, 'stopWords' => null, 'pattern' => '/[\p{L}\p{N}]+/u'], $tokenizer->exportConfig());
		$tokenizer->setMinLength(3);
		$tokenizer->setStopWords(['The', 'and']);
		$tokenizer->setPattern('/[a-z]+/');
		self::assertSame(['minLength' => 3, 'stopWords' => ['the', 'and'], 'pattern' => '/[a-z]+/'], $tokenizer->exportConfig());
	}

	public function testExportImportConfigRoundTrip()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setMinLength(3);
		$tokenizer->setStopWords(['the', 'and']);
		$tokenizer->setPattern('/[a-z]+/');
		$restored = new TWordTokenizer();
		$restored->importConfig($tokenizer->exportConfig());
		self::assertSame(3, $restored->getMinLength());
		self::assertSame(['the', 'and'], $restored->getStopWords());
		self::assertSame('/[a-z]+/', $restored->getPattern());
		self::assertSame($tokenizer->exportConfig(), $restored->exportConfig());
		self::assertSame($tokenizer->tokenize('The cat and the dog'), $restored->tokenize('The cat and the dog'));
	}

	public function testImportConfigWithNullStopWordsClearsThem()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->setStopWords(['the']);
		$tokenizer->importConfig(['stopWords' => null]);
		self::assertNull($tokenizer->getStopWords());
	}

	public function testImportConfigIgnoresUnknownAndPartialKeys()
	{
		$tokenizer = new TWordTokenizer();
		$tokenizer->importConfig(['minLength' => 5, 'bogus' => true]);
		self::assertSame(5, $tokenizer->getMinLength());
		self::assertNull($tokenizer->getStopWords());
		self::assertSame('/[\p{L}\p{N}]+/u', $tokenizer->getPattern());
	}

	public function testImportConfigRejectsInvalidPattern()
	{
		$tokenizer = new TWordTokenizer();
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$this->expectExceptionMessage('/(a+/');
		$tokenizer->importConfig(['pattern' => '/(a+/']);
	}
}
