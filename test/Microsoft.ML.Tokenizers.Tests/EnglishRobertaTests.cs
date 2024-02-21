﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class EnglishRobertaTests
    {
        private static readonly string _vocabUrl = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json";
        private static readonly string _mergeUrl = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe";
        private static readonly string _dictUrl = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt";

        public static IEnumerable<object[]> BertaData
        {
            get
            {
                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "Hello Berta",
                    new int[] { 15496, 22108, 64 },
                    new string[] { "Hello", "\u0120Bert", "a" },
                    new (int, int)[] { (0, 5), (5, 10), (10, 11) },
                    "Hello Berta",
                    new int[] { 35245, 144292, 18759122 },
                    new string[] { "Hello", " Bert", "a" },
                };

                // Intentionally repeating the same case data to test caching.
                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "Hello Berta",
                    new int[] { 15496, 22108, 64 },
                    new string[] { "Hello", "\u0120Bert", "a" },
                    new (int, int)[] { (0, 5), (5, 10), (10, 11) },
                    "Hello Berta",
                    new int[] { 35245, 144292, 18759122 },
                    new string[] { "Hello", " Bert", "a" },
                };

                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "In the night.", // Highest occurrence tokens
                    new int[] { 818, 262, 1755, 13 },
                    new string[] { "In", "\u0120the", "\u0120night", "." },
                    new (int, int)[] { (0, 2), (2, 6), (6, 12), (12, 13) },
                    "In the night.",
                    new int[] { 2224123, 800385005, 6062347, 850314647 },
                    new string[] { "In", " the", " night", "." },
                };

                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "He\U0001F601llo Ber\U0001F601ta", // Non-Latin characters should be ignored
                    new int[] { 1544, 18798, 4312, 8326 },
                    new string[] { "He", "llo", "ĠBer", "ta" },
                    new (int, int)[] { (0, 2), (4, 7), (7, 11), (13, 15) },
                    "Hello Berta",
                    new int[] { 2759525, 207306, 565286, 560191 },
                    new string[] { "He", "llo", " Ber", "ta" },
                };

                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "\U0001F601\U0001F601\u0660\u0340", // Full Non-Latin string
                    new int[] { },
                    new string[] {  },
                    new (int, int)[] { },
                    "",
                    new int[] {  },
                    new string[] {  },
                };
            }
        }

        [Fact]
        public async void TokenizationTest()
        {
            string vocabFile = Utils.CreateTemporaryFile("json");
            string mergeFile = Utils.CreateTemporaryFile("txt");
            string translationFile = Utils.CreateTemporaryFile("txt");
            string[]? paths = null; ;

            try
            {
                await Utils.DownloadFile(_vocabUrl, vocabFile);
                await Utils.DownloadFile(_mergeUrl, mergeFile);
                await Utils.DownloadFile(_dictUrl, translationFile);

                Tokenizer tokenizer = new Tokenizer(new EnglishRoberta(vocabFile, mergeFile, translationFile), RobertaPreTokenizer.Instance);
                TestTokenizer(tokenizer);

                using Stream vocabStream = File.OpenRead(vocabFile);
                using Stream mergeStream = File.OpenRead(mergeFile);
                using Stream translationStream = File.OpenRead(translationFile);
                tokenizer = new Tokenizer(new EnglishRoberta(vocabStream, mergeStream, translationStream), RobertaPreTokenizer.Instance);
                TestTokenizer(tokenizer);

                // Ensure caching works regardless of which method is called first.
                for (CallingOrder order = CallingOrder.Encode; order <= CallingOrder.CountTokens; order++)
                {
                    tokenizer = new Tokenizer(new EnglishRoberta(vocabFile, mergeFile, translationFile), RobertaPreTokenizer.Instance);
                    TestTokenizer(tokenizer, order);
                }
            }
            finally
            {
                Utils.DeleteFile(vocabFile);
                Utils.DeleteFile(mergeFile);
                Utils.DeleteFile(translationFile);

                if (paths is not null)
                {
                    Utils.DeleteFile(paths[0]);
                    Utils.DeleteFile(paths[1]);
                    Utils.DeleteFile(paths[2]);
                }
            }
        }

        private enum CallingOrder
        {
            Encode,
            EncodeToIds,
            CountTokens
        }

        // Calling EncodeToIds after calling Encode will cause EncodeToIds uses the cached data from the previous Encode call.
        // Calling with callIdsFirst = true will test the other way around.
        private void TestTokenizer(Tokenizer tokenizer, CallingOrder callingOrder = CallingOrder.Encode)
        {
            Assert.NotNull(tokenizer.Model);
            Assert.True(tokenizer.Model is EnglishRoberta);
            Assert.True(tokenizer.PreTokenizer is RobertaPreTokenizer);

            foreach (object[] p in BertaData)
            {
                IReadOnlyList<int> ids;
                TokenizerResult encoding;
                int idsCount;

                if (callingOrder == CallingOrder.Encode)
                {
                    encoding = tokenizer.Encode((string)p[0]);
                    ids = tokenizer.EncodeToIds((string)p[0]);
                    idsCount = tokenizer.CountTokens((string)p[0]);
                }
                else if (callingOrder == CallingOrder.EncodeToIds)
                {
                    ids = tokenizer.EncodeToIds((string)p[0]);
                    encoding = tokenizer.Encode((string)p[0]);
                    idsCount = tokenizer.CountTokens((string)p[0]);
                }
                else // CountTokens
                {
                    idsCount = tokenizer.CountTokens((string)p[0]);
                    ids = tokenizer.EncodeToIds((string)p[0]);
                    encoding = tokenizer.Encode((string)p[0]);
                }

                Assert.Equal(p[1], encoding.Ids);
                Assert.Equal(p[1], ids);
                Assert.Equal(((int[])p[1]).Length, idsCount);
                Assert.Equal(p[2], encoding.Tokens);
                Assert.Equal(p[3], encoding.Offsets);
                Assert.Equal(encoding.Ids.Count, encoding.Tokens.Count);
                Assert.Equal(encoding.Ids.Count, encoding.Offsets.Count);
                Assert.Equal(p[4], tokenizer.Decode(encoding.Ids));
                EnglishRoberta? robertaModel = tokenizer.Model as EnglishRoberta;
                Assert.NotNull(robertaModel);
                Assert.Equal(encoding.Ids, robertaModel!.OccurrenceRanksIds(robertaModel!.IdsToOccurrenceRanks(encoding.Ids)));
                Assert.Equal(p[5], robertaModel.IdsToOccurrenceValues(encoding.Ids));

                for (int i = 0; i < encoding.Tokens.Count; i++)
                {
                    Assert.Equal(encoding.Tokens[i], tokenizer.Model.IdToToken(encoding.Ids[i], considerSpecialTokens: false, filterUnsupportedChars: false));
                    Assert.Equal(encoding.Ids[i], tokenizer.Model.TokenToId(encoding.Tokens[i]));
                    Assert.Equal(encoding.Tokens[i], tokenizer.Decode(encoding.Ids[i], considerSpecialTokens: false, filterUnsupportedChars: false));

                    string[]? filteredToken = p[6] as string[];

                    Assert.Equal(filteredToken![i], tokenizer.Model.IdToToken(encoding.Ids[i], considerSpecialTokens: false, filterUnsupportedChars: true));
                }
            }
        }
    }
}
