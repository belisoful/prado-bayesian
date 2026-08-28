<?php

use Belisoful\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer;

class TNGramTokenizerTest extends PHPUnit\Framework\TestCase
{
	public function testCharacterTrigramsWithPadding()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(true);
		$tokens = $tokenizer->tokenize('hi');
		// "hi" has length 2 < n=3 with padding -> one padded n-gram of width 3.
		self::assertCount(1, $tokens);
		self::assertSame('hi ', $tokens[0]);
	}

	public function testCharacterTrigramsNoPaddingShortInput()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(false);
		self::assertSame([], $tokenizer->tokenize('hi'));
	}

	public function testCharacterTrigramsWithLongerInput()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(false);
		$tokens = $tokenizer->tokenize('hello');
		self::assertSame(['hel', 'ell', 'llo'], $tokens);
	}

	public function testCharacterTrigramsPaddedEmitsBoundaryNGrams()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(true);
		$tokens = $tokenizer->tokenize('abc');
		self::assertContains('a  ', $tokens);
		self::assertContains(' bc', $tokens);
		self::assertContains('abc', $tokens);
	}

	public function testUnigramsAreCharacters()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(1);
		$tokenizer->setCharacters(true);
		$tokens = $tokenizer->tokenize('abc');
		self::assertSame(['a', 'b', 'c'], $tokens);
	}

	public function testWordUnigrams()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(1);
		$tokenizer->setCharacters(false);
		$tokens = $tokenizer->tokenize('the quick brown fox');
		self::assertSame(['the', 'quick', 'brown', 'fox'], $tokens);
	}

	public function testWordBigrams()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(2);
		$tokenizer->setCharacters(false);
		$tokens = $tokenizer->tokenize('the quick brown fox');
		self::assertSame(['the quick', 'quick brown', 'brown fox'], $tokens);
	}

	public function testWordBigramsShortInputEmpty()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(false);
		self::assertSame([], $tokenizer->tokenize('one two'));
	}

	public function testEmptyInput()
	{
		$tokenizer = new TNGramTokenizer();
		self::assertSame([], $tokenizer->tokenize(''));
	}

	public function testNClampedToOne()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(0);
		self::assertSame(1, $tokenizer->getN());
	}

	public function testMultibyteCharacters()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(false);
		$tokens = $tokenizer->tokenize('café');
		self::assertSame(['caf', 'afé'], $tokens);
	}

	public function testPaddedCharacterNGramsWithMultibyteShortInput()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(true);
		// 'é' is 1 character (2 bytes); padded to width 3 it should be 'é  ' (right-padded).
		$tokens = $tokenizer->tokenize('é');
		self::assertCount(1, $tokens);
		self::assertSame(3, mb_strlen($tokens[0]));
		self::assertSame('é  ', $tokens[0]);
	}

	public function testPaddedCharacterNGramsWithMultibyteLongerInput()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);
		$tokenizer->setPad(true);
		// 'café' has 4 characters; padded trigrams should each be 3 characters.
		$tokens = $tokenizer->tokenize('café');
		foreach ($tokens as $token) {
			self::assertSame(3, mb_strlen($token), "Token '{$token}' should be 3 characters");
		}
	}

	public function testCharacterModeWhitespaceOnlyInputIsEmpty()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setCharacters(true);
		self::assertSame([], $tokenizer->tokenize('   '));
		self::assertSame([], $tokenizer->tokenize("\t\n "));
	}

	public function testCharacterModeTrimsSurroundingWhitespace()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setCharacters(true);
		self::assertSame($tokenizer->tokenize('ab'), $tokenizer->tokenize('  ab  '));
		self::assertNotSame([], $tokenizer->tokenize('  ab  '));
	}

	public function testCharacterModeCollapsesInternalWhitespace()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setCharacters(true);
		self::assertSame($tokenizer->tokenize('a b'), $tokenizer->tokenize('a   b'));
		self::assertSame($tokenizer->tokenize('a b'), $tokenizer->tokenize("a\t\nb"));
	}

	public function testExportConfigIncludesNestedWordTokenizer()
	{
		$tokenizer = new TNGramTokenizer();
		$config = $tokenizer->exportConfig();
		self::assertSame(3, $config['n']);
		self::assertTrue($config['characters']);
		self::assertTrue($config['pad']);
		self::assertSame($tokenizer->getWordTokenizer()->exportConfig(), $config['wordTokenizer']);
	}

	public function testExportImportConfigRoundTrip()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setN(2);
		$tokenizer->setCharacters(false);
		$tokenizer->setPad(false);
		$tokenizer->getWordTokenizer()->setMinLength(4);
		$tokenizer->getWordTokenizer()->setStopWords(['with']);
		$restored = new TNGramTokenizer();
		$restored->importConfig($tokenizer->exportConfig());
		self::assertSame(2, $restored->getN());
		self::assertFalse($restored->getCharacters());
		self::assertFalse($restored->getPad());
		self::assertSame(4, $restored->getWordTokenizer()->getMinLength());
		self::assertSame(['with'], $restored->getWordTokenizer()->getStopWords());
		self::assertSame($tokenizer->exportConfig(), $restored->exportConfig());
		$text = 'quick brown foxes with lazy dogs';
		self::assertSame($tokenizer->tokenize($text), $restored->tokenize($text));
		self::assertSame(['quick brown', 'brown foxes', 'foxes lazy', 'lazy dogs'], $restored->tokenize($text));
	}

	public function testImportConfigIgnoresMissingKeys()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->importConfig(['n' => 4]);
		self::assertSame(4, $tokenizer->getN());
		self::assertTrue($tokenizer->getCharacters());
		self::assertTrue($tokenizer->getPad());
		self::assertSame(2, $tokenizer->getWordTokenizer()->getMinLength());
	}

	public function testImportConfigIgnoresNonArrayWordTokenizer()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->importConfig(['wordTokenizer' => 'nope']);
		self::assertSame(2, $tokenizer->getWordTokenizer()->getMinLength());
	}

	public function testInvalidUtf8InputDoesNotThrow()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setCharacters(true);
		$tokens = $tokenizer->tokenize("caf\xE9 ok");
		self::assertNotSame([], $tokens);
		foreach ($tokens as $token) {
			self::assertTrue(mb_check_encoding($token, 'UTF-8'));
		}
	}

	public function testWordTokenizerIsReplaceable()
	{
		$tokenizer = new TNGramTokenizer();
		$tokenizer->setCharacters(false);
		$tokenizer->setN(2);
		self::assertInstanceOf(\Belisoful\Prado\Util\Bayesian\Tokenizer\TWordTokenizer::class, $tokenizer->getWordTokenizer());
		$inner = new \Belisoful\Prado\Util\Bayesian\Tokenizer\TWordTokenizer();
		// A minimum length of 4 drops the short words before the n-grams are built.
		$inner->setMinLength(4);
		$tokenizer->setWordTokenizer($inner);
		self::assertSame($inner, $tokenizer->getWordTokenizer());
		self::assertSame(['free offer'], $tokenizer->tokenize('a free offer to me'));
	}
}
