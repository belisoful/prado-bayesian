<?php

use Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerChain;
use Prado\Util\Bayesian\Tokenizer\TRegexTokenizer;
use Prado\Util\Bayesian\Tokenizer\TWordTokenizer;

class TBayesianTokenizerChainTest extends PHPUnit\Framework\TestCase
{
	public function testEmptyChainProducesEmpty()
	{
		$chain = new TBayesianTokenizerChain();
		self::assertSame([], $chain->tokenize('hello world'));
	}

	public function testChainConcatenatesTokenizers()
	{
		$chain = new TBayesianTokenizerChain();
		$word = new TWordTokenizer();
		$regex = new TRegexTokenizer();
		$regex->setPattern('/([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/');
		$chain->addTokenizer($word);
		$chain->addTokenizer($regex);
		$tokens = $chain->tokenize('Contact me at test@example.com today');
		self::assertContains('contact', $tokens);
		self::assertContains('me', $tokens);
		self::assertContains('at', $tokens);
		self::assertContains('today', $tokens);
		self::assertContains('test@example.com', $tokens);
	}

	public function testRemoveTokenizer()
	{
		$chain = new TBayesianTokenizerChain();
		$word = new TWordTokenizer();
		$chain->addTokenizer($word);
		self::assertTrue($chain->removeTokenizer($word));
		self::assertSame([], $chain->tokenize('hello'));
	}

	public function testRemoveTokenizerReturnsFalseWhenAbsent()
	{
		$chain = new TBayesianTokenizerChain();
		$other = new TWordTokenizer();
		self::assertFalse($chain->removeTokenizer($other));
	}

	public function testGetTokenizersReturnsInOrder()
	{
		$chain = new TBayesianTokenizerChain();
		$first = new TWordTokenizer();
		$second = new TRegexTokenizer();
		$chain->addTokenizer($first);
		$chain->addTokenizer($second);
		self::assertSame([$first, $second], $chain->getTokenizers());
	}

	public function testClear()
	{
		$chain = new TBayesianTokenizerChain();
		$chain->addTokenizer(new TWordTokenizer());
		$chain->addTokenizer(new TRegexTokenizer());
		$chain->clear();
		self::assertSame([], $chain->getTokenizers());
	}

	public function testExportConfigListsMembersWithClassAndConfig()
	{
		$chain = new TBayesianTokenizerChain();
		$word = new TWordTokenizer();
		$word->setMinLength(3);
		$regex = new TRegexTokenizer();
		$regex->setLowercase(false);
		$chain->addTokenizer($word);
		$chain->addTokenizer($regex);
		$config = $chain->exportConfig();
		self::assertCount(2, $config['tokenizers']);
		self::assertSame(TWordTokenizer::class, $config['tokenizers'][0]['class']);
		self::assertSame(3, $config['tokenizers'][0]['config']['minLength']);
		self::assertSame(TRegexTokenizer::class, $config['tokenizers'][1]['class']);
		self::assertFalse($config['tokenizers'][1]['config']['lowercase']);
	}

	public function testEmptyChainExportsEmptyMemberList()
	{
		$chain = new TBayesianTokenizerChain();
		self::assertSame(['tokenizers' => []], $chain->exportConfig());
	}

	public function testExportImportConfigRoundTrip()
	{
		$chain = new TBayesianTokenizerChain();
		$word = new TWordTokenizer();
		$word->setMinLength(3);
		$word->setStopWords(['the']);
		$ngram = new \Prado\Util\Bayesian\Tokenizer\TNGramTokenizer();
		$ngram->setN(2);
		$ngram->setCharacters(false);
		$chain->addTokenizer($word);
		$chain->addTokenizer($ngram);

		$restored = new TBayesianTokenizerChain();
		$restored->importConfig($chain->exportConfig());
		$members = $restored->getTokenizers();
		self::assertCount(2, $members);
		self::assertInstanceOf(TWordTokenizer::class, $members[0]);
		self::assertSame(3, $members[0]->getMinLength());
		self::assertSame(['the'], $members[0]->getStopWords());
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer::class, $members[1]);
		self::assertSame(2, $members[1]->getN());
		self::assertFalse($members[1]->getCharacters());
		self::assertSame($chain->exportConfig(), $restored->exportConfig());
		$text = 'the quick brown foxes';
		self::assertSame($chain->tokenize($text), $restored->tokenize($text));
	}

	public function testImportConfigReusesExistingMembersOfSameClass()
	{
		$chain = new TBayesianTokenizerChain();
		$word = new TWordTokenizer();
		$chain->addTokenizer($word);
		$chain->importConfig(['tokenizers' => [
			['class' => TWordTokenizer::class, 'config' => ['minLength' => 5]],
			['class' => TRegexTokenizer::class, 'config' => []],
		]]);
		$members = $chain->getTokenizers();
		self::assertCount(2, $members);
		self::assertSame($word, $members[0], 'a member of the same class at the same position is reconfigured in place');
		self::assertSame(5, $word->getMinLength());
		self::assertInstanceOf(TRegexTokenizer::class, $members[1]);
	}

	public function testImportConfigWithoutTokenizersKeyKeepsMembers()
	{
		$chain = new TBayesianTokenizerChain();
		$chain->addTokenizer(new TWordTokenizer());
		$chain->importConfig([]);
		self::assertCount(1, $chain->getTokenizers());
		$chain->importConfig(['tokenizers' => 'nope']);
		self::assertCount(1, $chain->getTokenizers());
	}

	public function testImportConfigSkipsNonArrayAndClasslessEntries()
	{
		$chain = new TBayesianTokenizerChain();
		$chain->importConfig(['tokenizers' => [
			'garbage',
			['config' => ['minLength' => 3]],
			['class' => TWordTokenizer::class, 'config' => []],
		]]);
		$members = $chain->getTokenizers();
		self::assertCount(1, $members);
		self::assertInstanceOf(TWordTokenizer::class, $members[0]);
	}

	public function testImportConfigRejectsNonTokenizerMemberClass()
	{
		$chain = new TBayesianTokenizerChain();
		try {
			$chain->importConfig(['tokenizers' => [['class' => \stdClass::class]]]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_class_invalid', $e->getErrorCode());
		}
	}

	public function testNestedChainRoundTrips()
	{
		$inner = new TBayesianTokenizerChain();
		$inner->addTokenizer(new TRegexTokenizer());
		$outer = new TBayesianTokenizerChain();
		$outer->addTokenizer(new TWordTokenizer());
		$outer->addTokenizer($inner);
		$restored = new TBayesianTokenizerChain();
		$restored->importConfig($outer->exportConfig());
		$members = $restored->getTokenizers();
		self::assertCount(2, $members);
		self::assertInstanceOf(TBayesianTokenizerChain::class, $members[1]);
		self::assertCount(1, $members[1]->getTokenizers());
		self::assertSame($outer->tokenize('Hello world'), $restored->tokenize('Hello world'));
	}
}
