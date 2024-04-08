// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the CodeGen Encoding model. This class is used to encode the text to a list of tokens.
    ///
    /// </summary>
    public sealed class CodeGen : Model
    {
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;
        private Dictionary<string, int>? _vocabOriginal;
        private readonly SortedDictionary<int, StringSpanOrdinalKey> _vocabReverse;
        private readonly Cache<(string, string), int> _mergeRanks;
        private readonly StringSpanOrdinalKeyCache<List<Token>> _cache;

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        public CodeGen(string vocabularyPath, string mergePath) :
            this(vocabularyPath is null ? throw new ArgumentNullException(nameof(vocabularyPath)) : File.OpenRead(vocabularyPath),
                 mergePath is null ? throw new ArgumentNullException(nameof(mergePath)) : File.OpenRead(mergePath), true)
        {
        }

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        public CodeGen(Stream vocabularyStream, Stream mergeStream) :
            this(vocabularyStream, mergeStream, false)
        {
        }

        private CodeGen(Stream vocabularyStream, Stream mergeStream, bool disposeStreams)
        {
            if (vocabularyStream is null)
            {
                throw new ArgumentNullException(nameof(vocabularyStream));
            }

            if (mergeStream is null)
            {
                throw new ArgumentNullException(nameof(mergeStream));
            }

            _vocab = Helpers.GetVocabulary(vocabularyStream);
            _vocabReverse = _vocab.ReverseSorted();
            _mergeRanks = GetMergeRanks(mergeStream);
            _cache = new StringSpanOrdinalKeyCache<List<Token>>();

            if (disposeStreams)
            {
                vocabularyStream.Dispose();
                mergeStream.Dispose();
            }
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocab => Helpers.GetVocab(_vocab, ref _vocabOriginal);

        //
        // Public Model interfaces implementation
        //

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the string.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id)
        {
            if (_vocabReverse.TryGetValue(id, out var value))
            {
                return value.Data!;
            }

            return null;
        }

        /// <summary>
        /// Encode a text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text)
        {
            if (text.IsEmpty)
            {
                return Bpe.EmptyTokensList;
            }

            char[] token = ArrayPool<char>.Shared.Rent(text.Length);
            int[] indexMapping = ArrayPool<int>.Shared.Rent(text.Length);

            int newTokenIndex = 0;
            IReadOnlyDictionary<char, char> byteToUnicode = ByteToUnicodeEncoding.Instance.ByteToUnicode;

            for (int i = 0; i < text.Length; i++)
            {
                if (byteToUnicode.TryGetValue(text[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                return Array.Empty<Token>();
            }

            if (_cache.TryGetValue(text, out List<Token>? hit))
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                return ModifyTokenListOffsets(hit, indexMapping);
            }

            List<Token> result = EncodeToTokens(token.AsSpan().Slice(0, newTokenIndex), indexMapping);
            _cache.Set(text.ToString(), result);
            ArrayPool<char>.Shared.Return(token);
            ArrayPool<int>.Shared.Return(indexMapping);
            return result;
        }

        /// <summary>
        /// Encode a split text string to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds, out int textLength, int maxTokens = int.MaxValue) => EncodeToIdsInternal(text, accumulatedIds, out textLength, maxTokens);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> text, out int textLength, int maxTokens = int.MaxValue) => EncodeToIdsInternal(text, null, out textLength, maxTokens);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textIndex">Starting from this index to the end of the text will encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokensFromEnd(ReadOnlySpan<char> text, out int textIndex, int maxTokens = int.MaxValue) => EncodeToIdsFromEndInternal(text, null, out textIndex, maxTokens);

        private int EncodeToIdsResult(List<Token> tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int textLength)
        {
            textLength = 0;

            if (tokens.Count <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                textLength = fullTextLength;
                return tokens.Count;
            }

            if (accumulatedIds is not null)
            {
                for (int i = 0; i < maxTokens; i++)
                {
                    accumulatedIds.Add(tokens[i].Id);
                    textLength += tokens[i].Offset.Length;
                }
            }
            else
            {
                for (int i = 0; i < maxTokens; i++)
                {
                    textLength += tokens[i].Offset.Length;
                }
            }

            return maxTokens;
        }

        private int EncodeToIdsFromEndResult(List<Token> tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int textIndex)
        {
            textIndex = fullTextLength;

            if (tokens.Count <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                textIndex = 0;
                return tokens.Count;
            }

            if (accumulatedIds is not null)
            {
                for (int i = tokens.Count - maxTokens; i < tokens.Count; i++)
                {
                    accumulatedIds.Add(tokens[i].Id);
                    textIndex -= tokens[i].Offset.Length;
                }
            }
            else
            {
                for (int i = tokens.Count - maxTokens; i < tokens.Count; i++)
                {
                    textIndex -= tokens[i].Offset.Length;
                }
            }

            return maxTokens;
        }

        private int EncodeToIdsInternal(ReadOnlySpan<char> text, IList<int>? accumulatedIds, out int textLength, int maxTokens)
        {
            if (text.IsEmpty)
            {
                textLength = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out List<Token>? hit))
            {
                return EncodeToIdsResult(hit, accumulatedIds, maxTokens, text.Length, out textLength);
            }

            char[] token = ArrayPool<char>.Shared.Rent(text.Length);
            int[] indexMapping = ArrayPool<int>.Shared.Rent(text.Length);

            int newTokenIndex = 0;
            IReadOnlyDictionary<char, char> byteToUnicode = ByteToUnicodeEncoding.Instance.ByteToUnicode;

            for (int i = 0; i < text.Length; i++)
            {
                if (byteToUnicode.TryGetValue(text[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                textLength = 0;
                return 0;
            }

            List<Token> result = EncodeToTokens(token.AsSpan().Slice(0, newTokenIndex), indexMapping);
            _cache.Set(text.ToString(), result);
            ArrayPool<char>.Shared.Return(token);
            ArrayPool<int>.Shared.Return(indexMapping);

            return EncodeToIdsResult(result, accumulatedIds, maxTokens, text.Length, out textLength);
        }

        private int EncodeToIdsFromEndInternal(ReadOnlySpan<char> text, IList<int>? accumulatedIds, out int textIndex, int maxTokens)
        {
            if (text.IsEmpty)
            {
                textIndex = text.Length;
                return 0;
            }

            if (_cache.TryGetValue(text, out List<Token>? hit))
            {
                return EncodeToIdsFromEndResult(hit, accumulatedIds, maxTokens, text.Length, out textIndex);
            }

            char[] token = ArrayPool<char>.Shared.Rent(text.Length);
            int[] indexMapping = ArrayPool<int>.Shared.Rent(text.Length);

            int newTokenIndex = 0;
            IReadOnlyDictionary<char, char> byteToUnicode = ByteToUnicodeEncoding.Instance.ByteToUnicode;

            for (int i = 0; i < text.Length; i++)
            {
                if (byteToUnicode.TryGetValue(text[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                ArrayPool<char>.Shared.Return(token);
                ArrayPool<int>.Shared.Return(indexMapping);
                textIndex = text.Length;
                return 0;
            }

            List<Token> result = EncodeToTokens(token.AsSpan().Slice(0, newTokenIndex), indexMapping);
            _cache.Set(text.ToString(), result);
            ArrayPool<char>.Shared.Return(token);
            ArrayPool<int>.Shared.Return(indexMapping);

            return EncodeToIdsFromEndResult(result, accumulatedIds, maxTokens, text.Length, out textIndex);
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token) => _vocab.TryGetValue(token, out int value) ? value : null;

        //
        // Private & Internal methods
        //

        private IReadOnlyList<Token> ModifyTokenListOffsets(IReadOnlyList<Token> tokens, Span<int> indexMapping)
        {
            int index = 0;

            for (int i = 0; i < tokens.Count; i++)
            {
                Debug.Assert(index + tokens[i].Value.Length <= indexMapping.Length);

                if (tokens[i].Offset != (indexMapping[index], tokens[i].Value.Length))
                {
                    List<Token> list = new List<Token>(tokens.Count);
                    for (int j = 0; j < i; j++)
                    {
                        list.Add(tokens[j]);
                    }

                    for (int j = i; j < tokens.Count; j++)
                    {
                        list.Add(new Token(tokens[j].Id, tokens[j].Value, (indexMapping[index], tokens[j].Value.Length)));
                        index += tokens[j].Value.Length;
                    }

                    return list;
                }

                index += tokens[i].Value.Length;
            }

            return tokens;
        }

        private Cache<(string, string), int> GetMergeRanks(Stream mergeStream)
        {
            var mergeRanks = new Cache<(string, string), int>(60_000);
            try
            {
                using StreamReader reader = new StreamReader(mergeStream);

                // We ignore the first and last line in the file
                if (reader.Peek() >= 0)
                {
                    string ignored = reader.ReadLine()!;
                }

                int rank = 1;
                while (reader.Peek() >= 0)
                {
                    string line = reader.ReadLine()!;
                    int index = line.IndexOf(' ');
                    if (index < 1 || index == line.Length - 1 || line.IndexOf(' ', index + 1) != -1)
                    {
                        throw new Exception($"Invalid format of merge file: \"{line}\"");
                    }

                    mergeRanks.Set((line.Substring(0, index), line.Substring(index + 1)), rank++);
                }
            }
            catch (Exception e)
            {
                throw new IOException($"Cannot read the file Merge file.{Environment.NewLine}Error message: {e.Message}", e);
            }

            return mergeRanks;
        }

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private List<Token> EncodeToTokens(Span<char> token, Span<int> indexMapping)
        {
            if (token.Length == 0)
            {
                return Bpe.EmptyTokensList;
            }

            string[] charToString = ByteToUnicodeEncoding.Instance.CharToString;

            if (token.Length == 1)
            {
                string tokenValue = charToString[token[0]];
                return new List<Token> { new Token(_vocab[new StringSpanOrdinalKey(tokenValue)], tokenValue, (indexMapping[0], 1)) };
            }

            List<string> word = new(token.Length);
            foreach (char c in token)
            {
                Debug.Assert(c < charToString.Length);
                word.Add(charToString[c]);
            }

            HashSet<(string, string)> pairs = new();

            WordToPairs(word, pairs);

            var newWord = new List<string>();

            Debug.Assert(pairs.Count != 0, "Pairs should not be empty.");

            while (true)
            {
                /* while conditions */
                // if only one element left, merge is finished (with the whole word merged)
                if (word.Count == 1)
                {
                    break;
                }

                // get the most frequent bi-gram pair
                var (first, second) = pairs.ArgMin(pair => _mergeRanks.GetOrAdd(pair, int.MaxValue));
                if (!_mergeRanks.TryGetValue((first, second), out int _))
                {
                    break;
                }
                /* end while conditions */

                // search and merge all (first, second) pairs in {word}
                var i = 0;
                while (i < word.Count)
                {
                    // find the next occurrence of {first} and add the elements before into {newWord}
                    var j = word.IndexOf(first, i);
                    if (j == -1)
                    {
                        // Equivalent to newWord.AddRange(word.Skip(i)) without allocations
                        for (int k = i; k < word.Count; k++)
                        {
                            newWord.Add(word[k]);
                        }

                        break;
                    }
                    else
                    {
                        // Equivalent to newWord.AddRange(word.Skip(i).Take(j - i)) without allocations
                        for (int k = i; k < j; k++)
                        {
                            newWord.Add(word[k]);
                        }

                        i = j;
                    }

                    // check the next element is {second} or not
                    if (i < word.Count - 1 && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }

                List<string> temp = word;
                word = newWord;
                newWord = temp;
                newWord.Clear();

                // otherwise, continue merging
                WordToPairs(word, pairs);
            }

            var tokens = new List<Token>(word.Count);
            int index = 0;

            foreach (string w in word)
            {
                tokens.Add(new Token(_vocab[new StringSpanOrdinalKey(w)], w, (indexMapping[index], w.Length)));
                index += w.Length;
            }

            return tokens;
        }

        /// <summary>
        /// Extract element pairs in an aggregating word. E.g. [p, l, ay] into [(p,l), (l,ay)].
        /// If word contains 0 or 1 element, an empty HashSet will be returned.
        /// </summary>
        private static void WordToPairs(IReadOnlyList<string> word, HashSet<(string, string)> pairs)
        {
            pairs.Clear();

            if (word.Count <= 1)
            {
                return;
            }

            var prevElem = word[0];
            foreach (var elem in word.Skip(1))
            {
                pairs.Add((prevElem, elem));
                prevElem = elem;
            }
        }
    }
}
