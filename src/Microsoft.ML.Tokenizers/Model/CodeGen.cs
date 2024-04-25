﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// Implement the CodeGen tokenizer described in https://huggingface.co/docs/transformers/main/en/model_doc/codegen#overview
    /// </summary>
    public sealed class CodeGen : Tokenizer
    {
        private readonly Dictionary<StringSpanOrdinalKey, (int Id, string Token)> _vocab;
        private Dictionary<string, int>? _vocabOriginal;
        private readonly Dictionary<int, string> _vocabReverse;
        private readonly Dictionary<StringSpanOrdinalKey, (int, string)>? _addedTokens;
        private readonly Dictionary<int, string>? _addedTokensReverse;
        private readonly Dictionary<StringSpanOrdinalKeyPair, int> _mergeRanks;
        private readonly StringSpanOrdinalKeyCache<List<Token>> _cache;
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;
        private const int MaxTokenLengthToCache = 15;
        private const string DefaultSpecialToken = "<|endoftext|>";

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="addedTokens">The additional tokens to add to the vocabulary.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="unknownToken">The unknown token.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        public CodeGen(
                string vocabularyPath,
                string mergePath,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                IReadOnlyDictionary<string, int>? addedTokens = null,
                bool addPrefixSpace = false,
                bool addBeginningOfSentence = false,
                bool addEndOfSentence = false,
                string? unknownToken = DefaultSpecialToken,
                string? beginningOfSentenceToken = DefaultSpecialToken,
                string? endOfSentenceToken = DefaultSpecialToken) :
            this(vocabularyPath is null ? throw new ArgumentNullException(nameof(vocabularyPath)) : File.OpenRead(vocabularyPath),
                mergePath is null ? throw new ArgumentNullException(nameof(mergePath)) : File.OpenRead(mergePath),
                preTokenizer, normalizer, addedTokens, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, unknownToken, beginningOfSentenceToken, endOfSentenceToken, disposeStream: true)
        {
        }

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="addedTokens">The additional tokens to add to the vocabulary.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="unknownToken">The unknown token.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        public CodeGen(
                Stream vocabularyStream,
                Stream mergeStream,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                IReadOnlyDictionary<string, int>? addedTokens = null,
                bool addPrefixSpace = false,
                bool addBeginningOfSentence = false,
                bool addEndOfSentence = false,
                string? unknownToken = DefaultSpecialToken,
                string? beginningOfSentenceToken = DefaultSpecialToken,
                string? endOfSentenceToken = DefaultSpecialToken) :
            this(vocabularyStream, mergeStream, preTokenizer, normalizer, addedTokens, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, unknownToken, beginningOfSentenceToken, endOfSentenceToken, disposeStream: false)
        {
        }

        private CodeGen(Stream vocabularyStream, Stream mergeStream, PreTokenizer? preTokenizer, Normalizer? normalizer, IReadOnlyDictionary<string, int>? addedTokens, bool addPrefixSpace,
                        bool addBeginningOfSentence, bool addEndOfSentence, string? unknownToken, string? beginningOfSentenceToken, string? endOfSentenceToken, bool disposeStream)
        {
            if (vocabularyStream is null)
            {
                throw new ArgumentNullException(nameof(vocabularyStream));
            }

            if (mergeStream is null)
            {
                throw new ArgumentNullException(nameof(mergeStream));
            }

            _preTokenizer = preTokenizer;
            _normalizer = normalizer;

            // Tokenizer data files can be found in codegen-350M-mono
            // https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json?download=true
            // https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt?download=true

            // Or in Phi-2
            // https://huggingface.co/microsoft/phi-2/resolve/main/vocab.json?download=true
            // https://huggingface.co/microsoft/phi-2/resolve/main/merges.txt?download=true

            _vocab = GetVocabulary(vocabularyStream);
            _vocabReverse = _vocab.ToDictionary(kvp => kvp.Value.Id, kvp => kvp.Value.Token);
            _mergeRanks = GetMergeRanks(mergeStream);
            _cache = new StringSpanOrdinalKeyCache<List<Token>>();

            if (addedTokens is not null)
            {
                AddedTokens = addedTokens;
                _addedTokens = addedTokens.ToDictionary(kvp => new StringSpanOrdinalKey(kvp.Key), kvp => (kvp.Value, kvp.Key));
                _addedTokensReverse = addedTokens.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            }

            UnknownToken = unknownToken;
            BeginningOfSentenceToken = beginningOfSentenceToken;
            EndOfSentenceToken = endOfSentenceToken;

            AddPrefixSpace = addPrefixSpace;
            AddBeginningOfSentence = addBeginningOfSentence;
            AddEndOfSentence = addEndOfSentence;

            if (!string.IsNullOrEmpty(UnknownToken))
            {
                if (!_vocab.TryGetValue(UnknownToken!, out (int unknownId, string token) value))
                {
                    throw new ArgumentException($"The Unknown token '{UnknownToken}' is not found in the vocabulary.");
                }

                UnknownTokenId = value.unknownId;
            }

            if (!string.IsNullOrEmpty(BeginningOfSentenceToken))
            {
                if (!_vocab.TryGetValue(BeginningOfSentenceToken!, out (int beggingOfSentenceId, string token) value))
                {
                    throw new ArgumentException($"The beginning of sentence token '{BeginningOfSentenceToken}' is not found in the vocabulary.");
                }

                BeginningOfSentenceId = value.beggingOfSentenceId;
            }

            if (!string.IsNullOrEmpty(EndOfSentenceToken))
            {
                if (!_vocab.TryGetValue(EndOfSentenceToken!, out (int endOfSentenceId, string token) value))
                {
                    throw new ArgumentException($"The end of sentence token '{EndOfSentenceToken}' is not found in the vocabulary.");
                }

                EndOfSentenceId = value.endOfSentenceId;
            }

            if (AddBeginningOfSentence && string.IsNullOrEmpty(BeginningOfSentenceToken))
            {
                throw new ArgumentException("The beginning of sentence token must be provided when the flag is set to include it in the encoding.");
            }

            if (AddEndOfSentence && string.IsNullOrEmpty(EndOfSentenceToken))
            {
                throw new ArgumentException("The end of sentence token must be provided when the flag is set to include it in the encoding.");
            }

            if (disposeStream)
            {
                vocabularyStream.Dispose();
                mergeStream.Dispose();
            }
        }

        /// <summary>
        /// Gets the added tokens.
        /// </summary>
        public IReadOnlyDictionary<string, int>? AddedTokens { get; }

        /// <summary>
        /// The Unknown token.
        /// </summary>
        public string? UnknownToken { get; }

        /// <summary>
        /// Gets the Unknown token Id.
        /// </summary>
        public int? UnknownTokenId { get; }

        /// <summary>
        /// Gets the flag indicating whether to include the beginning of sentence token in the encoding.
        /// </summary>
        public bool AddBeginningOfSentence { get; }

        /// <summary>
        /// Gets the flag indicating whether to include the end of sentence token in the encoding.
        /// </summary>
        public bool AddEndOfSentence { get; }

        /// <summary>
        /// Gets the beginning of sentence token.
        /// </summary>
        public string? BeginningOfSentenceToken { get; }

        /// <summary>
        /// Gets the end of sentence token Id.
        /// </summary>
        public int? BeginningOfSentenceId { get; }

        /// <summary>
        /// Gets the end of sentence token Id.
        /// </summary>
        public int? EndOfSentenceId { get; }

        /// <summary>
        /// Gets the end of sentence token.
        /// </summary>
        public string? EndOfSentenceToken { get; }

        /// <summary>
        /// Gets the flag indicating whether to include a leading space before encoding the text.
        /// </summary>
        public bool AddPrefixSpace { get; }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => _preTokenizer;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocab
        {
            get
            {
                Dictionary<string, int>? publicVocab = Volatile.Read(ref _vocabOriginal);
                if (publicVocab is null)
                {
                    var vocab = _vocab.ToDictionary(kvp => kvp.Value.Token, kvp => kvp.Value.Id);
                    Interlocked.CompareExchange(ref _vocabOriginal, vocab, null);
                    publicVocab = _vocabOriginal;
                }

                return publicVocab;
            }
        }

        //
        // Public Model interfaces implementation
        //

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public override IReadOnlyList<Token> Encode(string text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
            => Encode(text, Span<char>.Empty, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, out normalizedString, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
            => Encode(null, text, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, out normalizedString, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will null.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<Token> Encode(string text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
            => Encode(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, out normalizedString, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will null.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
            => Encode(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, out normalizedString, considerPreTokenization, considerNormalization);

        private IReadOnlyList<Token> Encode(string? text, scoped ReadOnlySpan<char> textSpan, bool addPrefixSpace, bool addBos, bool addEos, out string? normalizedString, bool considerPreTokenization, bool considerNormalization)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return [];
            }

            const int bufferLength = 128;
            char[]? mutatedInputText = null;
            Span<char> mutatedInputSpan = stackalloc char[bufferLength];
            scoped ReadOnlySpan<char> textSpanToEncode;
            IEnumerable<(int Offset, int Length)>? splits;
            if (addPrefixSpace)
            {
                ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                if (span.Length + 1 > bufferLength)
                {
                    mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                    mutatedInputSpan = mutatedInputText;
                }
                mutatedInputSpan[0] = ' ';
                span.CopyTo(mutatedInputSpan.Slice(1));
                span = mutatedInputSpan.Slice(0, span.Length + 1);

                splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }
            else
            {
                splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }

            List<Token> tokens = new();
            if (addBos && BeginningOfSentenceId.HasValue)
            {
                tokens.Add(new Token(BeginningOfSentenceId.Value, BeginningOfSentenceToken!, (0, 0)));
            }

            PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), tokens, addPrefixSpace, split.Offset, agenda);
                }
            }
            else
            {
                EncodeInternal(addPrefixSpace ? null : (normalizedString ?? text), textSpanToEncode, tokens, addPrefixSpace, 0, agenda);
            }

            if (addEos && EndOfSentenceId.HasValue)
            {
                tokens.Add(new Token(EndOfSentenceId.Value, EndOfSentenceToken!, (addPrefixSpace ? Math.Max(0, textSpanToEncode.Length - 1) : textSpanToEncode.Length, 0)));
            }

            if (mutatedInputText is not null)
            {
                ArrayPool<char>.Shared.Return(mutatedInputText);
            }

            return tokens;
        }

        /// <summary>
        /// Encode a text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text in form of string to encode if it is available.</param>
        /// <param name="textSpan">The text in form of span to encode.</param>
        /// <param name="tokens">The tokens to include in the newly encoded sequence.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="offset">The offset to adjust the token's offset.</param>
        /// <param name="agenda">The priority queue to use for encoding.</param>
        private void EncodeInternal(string? text, scoped ReadOnlySpan<char> textSpan, List<Token> tokens, bool addPrefixSpace, int offset, PriorityQueue<SymbolPair> agenda)
        {
            if (textSpan.IsEmpty)
            {
                return;
            }

            if (_addedTokens is not null && _addedTokens.TryGetValue(textSpan, out (int addedTokenId, string addedToken) value))
            {
                tokens.Add(new Token(value.addedTokenId, value.addedToken, ((addPrefixSpace && offset > 0) ? offset - 1 : offset, (addPrefixSpace && offset == 0) ? textSpan.Length - 1 : textSpan.Length)));
                return;
            }

            if (_cache.TryGetValue(textSpan, out List<Token>? hit))
            {
                AppendTokenWithOffsetAdjusting(hit, tokens, offset, addPrefixSpace);
                return;
            }

            const int bufferLength = 64;
            Span<char> token = stackalloc char[bufferLength];
            Span<int> mapping = stackalloc int[bufferLength];
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(textSpan.Length);
            if (destinationMaxSize > bufferLength)
            {
                tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                token = tokenBuffer;

                mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                mapping = mappingBuffer;
            }

            int encodedLength = Helpers.EncodeToUtf8AndTransform(textSpan, token, mapping);

            List<Token> result = EncodeToTokens(token.Slice(0, encodedLength), mapping.Slice(0, encodedLength), textSpan, agenda);

            if (textSpan.Length <= MaxTokenLengthToCache)
            {
                _cache.Set(text is null ? textSpan.ToString() : text, result);
            }

            if (tokenBuffer is not null)
            {
                ArrayPool<char>.Shared.Return(tokenBuffer);
                Debug.Assert(mappingBuffer is not null);
                ArrayPool<int>.Shared.Return(mappingBuffer);
            }

            AppendTokenWithOffsetAdjusting(result, tokens, offset, addPrefixSpace);
        }

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        private IReadOnlyList<int> EncodeToIds(
                                    string? text,
                                    scoped ReadOnlySpan<char> textSpan,
                                    bool addPrefixSpace,
                                    bool addBeginningOfSentence,
                                    bool addEndOfSentence,
                                    bool considerPreTokenization,
                                    bool considerNormalization,
                                    out string? normalizedString,
                                    out int textLength,
                                    int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                textLength = 0;
                normalizedString = null;
                return [];
            }

            const int bufferLength = 128;
            char[]? mutatedInputText = null;
            Span<char> mutatedInputSpan = stackalloc char[bufferLength];
            scoped ReadOnlySpan<char> textSpanToEncode;
            IEnumerable<(int Offset, int Length)>? splits;
            if (addPrefixSpace)
            {
                ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                if (span.Length + 1 > bufferLength)
                {
                    mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                    mutatedInputSpan = mutatedInputText;
                }
                mutatedInputSpan[0] = ' ';
                span.CopyTo(mutatedInputSpan.Slice(1));
                span = mutatedInputSpan.Slice(0, span.Length + 1);

                splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }
            else
            {
                splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }

            List<int> ids = new();

            if (addBeginningOfSentence && BeginningOfSentenceId.HasValue)
            {
                ids.Add(BeginningOfSentenceId.Value);
            }

            PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

            if (splits is not null)
            {
                textLength = 0;
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToIdsInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), ids, agenda, out int length, maxTokenCount - ids.Count);
                    textLength = split.Offset + length;

                    if (length < split.Length || ids.Count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                EncodeToIdsInternal(addPrefixSpace ? null : (normalizedString ?? text), textSpanToEncode, ids, agenda, out textLength, maxTokenCount - ids.Count);
            }

            if (mutatedInputText is not null)
            {
                ArrayPool<char>.Shared.Return(mutatedInputText);
            }

            if (addEndOfSentence && EndOfSentenceId.HasValue && ids.Count < maxTokenCount)
            {
                ids.Add(EndOfSentenceId.Value);
            }

            if (addPrefixSpace && textLength > 0)
            {
                textLength--;
            }

            return ids;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        public override int CountTokens(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, Span<char>.Empty, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public override int IndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(text, Span<char>.Empty, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public override int IndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(null, text, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public int IndexOfTokenCount(string text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public int IndexOfTokenCount(
                    ReadOnlySpan<char> text,
                    int maxTokenCount,
                    bool addPrefixSpace,
                    bool addBeginningOfSentence,
                    bool addEndOfSentence,
                    out string? normalizedString,
                    out int tokenCount,
                    bool considerPreTokenization = true,
                    bool considerNormalization = true)
        {
            tokenCount = CountTokens(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        private int CountTokens(
                        string? text,
                        scoped ReadOnlySpan<char> textSpan,
                        bool addPrefixSpace,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerPreTokenization,
                        bool considerNormalization,
                        out string? normalizedString,
                        out int textLength,
                        int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            textLength = 0;
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return 0;
            }

            const int bufferLength = 128;
            char[]? mutatedInputText = null;
            Span<char> mutatedInputSpan = stackalloc char[bufferLength];
            scoped ReadOnlySpan<char> textSpanToEncode;
            IEnumerable<(int Offset, int Length)>? splits;
            if (addPrefixSpace)
            {
                ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                if (span.Length + 1 > bufferLength)
                {
                    mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                    mutatedInputSpan = mutatedInputText;
                }
                mutatedInputSpan[0] = ' ';
                span.CopyTo(mutatedInputSpan.Slice(1));
                span = mutatedInputSpan.Slice(0, span.Length + 1);

                splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }
            else
            {
                splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }

            PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

            int count = (addBeginningOfSentence && BeginningOfSentenceId.HasValue) ? 1 : 0;
            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    count += EncodeToIdsInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), null, agenda, out int length, maxTokenCount - count);
                    textLength = split.Offset + length;

                    if (length < split.Length || count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                count = EncodeToIdsInternal(addPrefixSpace ? null : text, textSpanToEncode, null, agenda, out textLength, maxTokenCount - count);
            }

            if (mutatedInputText is not null)
            {
                ArrayPool<char>.Shared.Return(mutatedInputText);
            }

            if (addEndOfSentence && EndOfSentenceId.HasValue && count < maxTokenCount)
            {
                count++;
            }

            if (addPrefixSpace && textLength > 0)
            {
                textLength--;
            }

            return count;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if normalization is enabled;
        /// conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public override int LastIndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(text, Span<char>.Empty, maxTokenCount, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedString"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public override int LastIndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(null, text, maxTokenCount, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if normalization is enabled;
        /// conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public int LastIndexOfTokenCount(string text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(text, Span<char>.Empty, maxTokenCount, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedString"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public int LastIndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(null, text, maxTokenCount, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        private int LastIndexOf(
                        string? text,
                        scoped ReadOnlySpan<char> textSpan,
                        int maxTokenCount,
                        bool addPrefixSpace,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerPreTokenization,
                        bool considerNormalization,
                        out string? normalizedString,
                        out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                tokenCount = 0;
                return 0;
            }

            const int bufferLength = 128;
            char[]? mutatedInputText = null;
            Span<char> mutatedInputSpan = stackalloc char[bufferLength];
            scoped ReadOnlySpan<char> textSpanToEncode;
            IEnumerable<(int Offset, int Length)>? splits;
            if (addPrefixSpace)
            {
                ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                if (span.Length + 1 > bufferLength)
                {
                    mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                    mutatedInputSpan = mutatedInputText;
                }
                mutatedInputSpan[0] = ' ';
                span.CopyTo(mutatedInputSpan.Slice(1));
                span = mutatedInputSpan.Slice(0, span.Length + 1);

                splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }
            else
            {
                splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out textSpanToEncode);
            }

            PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

            tokenCount = (addEndOfSentence && EndOfSentenceId.HasValue) ? 1 : 0;

            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits.Reverse())
                {
                    tokenCount += EncodeToIdsFromEndInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), null, agenda, out int textIndex, maxTokenCount - tokenCount);
                    if (textIndex > 0 || tokenCount >= maxTokenCount)
                    {
                        if (mutatedInputText is not null)
                        {
                            ArrayPool<char>.Shared.Return(mutatedInputText);
                        }
                        return addPrefixSpace ? split.Offset + textIndex - 1 : split.Offset + textIndex;
                    }
                }
            }
            else
            {
                tokenCount = EncodeToIdsFromEndInternal(addPrefixSpace ? null : text, textSpanToEncode, null, agenda, out int textLength, maxTokenCount - tokenCount);
                if (mutatedInputText is not null)
                {
                    ArrayPool<char>.Shared.Return(mutatedInputText);
                }

                return addPrefixSpace ? Math.Max(0, textLength - 1) : textLength;
            }

            if (mutatedInputText is not null)
            {
                ArrayPool<char>.Shared.Return(mutatedInputText);
            }

            if (addBeginningOfSentence && BeginningOfSentenceId.HasValue && tokenCount < maxTokenCount)
            {
                tokenCount++;
            }

            return 0;
        }

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

            int tokenCount;
            for (tokenCount = 0; tokenCount < maxTokens; tokenCount++)
            {
                // maxTokens is less than tokens.Count, so it is safe to
                if (tokens[tokenCount].Offset.Index == tokens[tokenCount + 1].Offset.Index)
                {
                    // Ensure we'll not break the text in the middle of a code-point
                    int j = tokenCount + 2;
                    while (j < tokens.Count && tokens[j].Offset.Index == tokens[tokenCount].Offset.Index)
                    {
                        j++;
                    }

                    if (j <= maxTokens)
                    {
                        // append encountered tokens to the accumulatedIds
                        for (int k = tokenCount; k < j; k++)
                        {
                            accumulatedIds?.Add(tokens[k].Id);
                            textLength += tokens[k].Offset.Length;
                        }
                        tokenCount = j - 1;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    accumulatedIds?.Add(tokens[tokenCount].Id);
                    textLength += tokens[tokenCount].Offset.Length;
                }
            }

            return tokenCount;
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

            int index = tokens.Count - maxTokens;
            if (index > 0)
            {
                // avoid breaking the text in the middle of a code-point
                while (index < tokens.Count && tokens[index].Offset.Index == tokens[index - 1].Offset.Index)
                {
                    index++;
                }
            }

            for (int i = index; i < tokens.Count; i++)
            {
                accumulatedIds?.Add(tokens[i].Id);
                textIndex -= tokens[i].Offset.Length;
            }

            return tokens.Count - index;
        }

        private int EncodeToIdsInternal(string? text, scoped ReadOnlySpan<char> textSpan, IList<int>? accumulatedIds, PriorityQueue<SymbolPair> agenda, out int textLength, int maxTokens)
        {
            if (textSpan.IsEmpty)
            {
                textLength = 0;
                return 0;
            }

            const int bufferLength = 100;

            if (_addedTokens is not null && _addedTokens.TryGetValue(textSpan, out (int addedTokenId, string addedToken) value) && maxTokens > 0)
            {
                if (accumulatedIds is not null)
                {
                    accumulatedIds.Add(value.addedTokenId);
                }

                textLength = textSpan.Length;
                return 1;
            }

            if (_cache.TryGetValue(textSpan, out List<Token>? hit))
            {
                return EncodeToIdsResult(hit, accumulatedIds, maxTokens, textSpan.Length, out textLength);
            }

            Span<char> token = stackalloc char[bufferLength];
            Span<int> mapping = stackalloc int[bufferLength];
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(textSpan.Length);
            if (destinationMaxSize > token.Length)
            {
                tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                token = tokenBuffer;

                mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                mapping = mappingBuffer;
            }

            int encodedLength = Helpers.EncodeToUtf8AndTransform(textSpan, token, mapping);

            List<Token> result = EncodeToTokens(token.Slice(0, encodedLength), mapping.Slice(0, encodedLength), textSpan, agenda);

            int length = text is not null ? text.Length : textSpan.Length;
            if (length <= MaxTokenLengthToCache)
            {
                _cache.Set(text ?? textSpan.ToString(), result);
            }

            if (tokenBuffer is not null)
            {
                ArrayPool<char>.Shared.Return(tokenBuffer);
                Debug.Assert(mappingBuffer is not null);
                ArrayPool<int>.Shared.Return(mappingBuffer);
            }

            return EncodeToIdsResult(result, accumulatedIds, maxTokens, textSpan.Length, out textLength);
        }

        private int EncodeToIdsFromEndInternal(string? text, scoped ReadOnlySpan<char> textSpan, IList<int>? accumulatedIds, PriorityQueue<SymbolPair> agenda, out int textIndex, int maxTokens)
        {
            if (textSpan.IsEmpty)
            {
                textIndex = textSpan.Length;
                return 0;
            }

            if (_addedTokens is not null && _addedTokens.TryGetValue(textSpan, out (int addedTokenId, string addedToken) value) && maxTokens > 0)
            {
                if (accumulatedIds is not null)
                {
                    accumulatedIds.Add(value.addedTokenId);
                }

                textIndex = 0;
                return 1;
            }

            if (_cache.TryGetValue(textSpan, out List<Token>? hit))
            {
                return EncodeToIdsFromEndResult(hit, accumulatedIds, maxTokens, textSpan.Length, out textIndex);
            }

            Span<char> token = stackalloc char[100];
            Span<int> mapping = stackalloc int[100];
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(textSpan.Length);
            if (destinationMaxSize > token.Length)
            {
                tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                token = tokenBuffer;

                mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                mapping = mappingBuffer;
            }

            int encodedLength = Helpers.EncodeToUtf8AndTransform(textSpan, token, mapping);

            List<Token> result = EncodeToTokens(token.Slice(0, encodedLength), mapping.Slice(0, encodedLength), textSpan, agenda);

            int length = text is not null ? text.Length : textSpan.Length;
            if (length <= MaxTokenLengthToCache)
            {
                _cache.Set(text ?? textSpan.ToString(), result);
            }

            if (tokenBuffer is not null)
            {
                ArrayPool<char>.Shared.Return(tokenBuffer);
                Debug.Assert(mappingBuffer is not null);
                ArrayPool<int>.Shared.Return(mappingBuffer);
            }

            return EncodeToIdsFromEndResult(result, accumulatedIds, maxTokens, textSpan.Length, out textIndex);
        }

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the string.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id)
        {
            if (_vocabReverse.TryGetValue(id, out var value))
            {
                return value;
            }

            if (_addedTokensReverse is not null && _addedTokensReverse.TryGetValue(id, out value))
            {
                return value;
            }

            return null;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token)
        {
            if (_vocab.TryGetValue(token, out (int Id, string Token) value))
            {
                return value.Id;
            }

            if (_addedTokens is not null && _addedTokens.TryGetValue(token, out (int Id, string Token) addedToken))
            {
                return addedToken.Id;
            }

            return null;
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string? Decode(IEnumerable<int> ids) => Decode(ids, hasPrefixSpace: AddPrefixSpace, considerSpecialTokens: false);

        private static readonly char _transformedSpace = ByteToUnicodeEncoding.Instance.ByteToUnicode[' '];

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="hasPrefixSpace">Indicate whether the encoded string has a leading space.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <returns>The decoded string.</returns>
        public string? Decode(IEnumerable<int> ids, bool hasPrefixSpace, bool considerSpecialTokens)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            byte[] bytes = ArrayPool<byte>.Shared.Rent(128);
            int bytesIndex = 0;
            bool firstToken = true;

            foreach (int id in ids)
            {
                if (BeginningOfSentenceId.HasValue && id == BeginningOfSentenceId.Value)
                {
                    if (considerSpecialTokens)
                    {
                        AppendToBytesArray(BeginningOfSentenceToken!.AsSpan(), ref bytes, ref bytesIndex);
                    }
                    continue;
                }

                if (EndOfSentenceId.HasValue && id == EndOfSentenceId.Value)
                {
                    if (considerSpecialTokens)
                    {
                        AppendToBytesArray(EndOfSentenceToken!.AsSpan(), ref bytes, ref bytesIndex);
                    }
                    continue;
                }

                if (UnknownTokenId.HasValue && id == UnknownTokenId.Value)
                {
                    if (considerSpecialTokens)
                    {
                        AppendToBytesArray(UnknownToken!.AsSpan(), ref bytes, ref bytesIndex);
                    }
                    continue;
                }

                if (_addedTokensReverse is not null && _addedTokensReverse.TryGetValue(id, out string? addedToken))
                {
                    int bytesCountToEncode = Encoding.UTF8.GetMaxByteCount(addedToken.Length);
                    if (bytes.Length - bytesIndex < bytesCountToEncode)
                    {
                        Helpers.ArrayPoolGrow(ref bytes, (bytes.Length + bytesCountToEncode) * 2);
                    }

                    bool removePrefixSpace = firstToken && hasPrefixSpace && addedToken.Length > 0 && addedToken[0] == ' ';
                    bytesIndex += Helpers.GetUtf8Bytes(removePrefixSpace ? addedToken.AsSpan().Slice(1) : addedToken.AsSpan(), bytes.AsSpan().Slice(bytesIndex));
                    firstToken = false;
                    continue;
                }

                // vocabularies are stored in UTF-8 form with escaping the control characters.
                // Need to convert the vocabulary to the original UTF-16 form.

                if (MapIdToToken(id) is string s)
                {
                    ReadOnlySpan<char> span = firstToken && hasPrefixSpace && s.Length > 0 && s[0] == _transformedSpace ? s.AsSpan(1) : s.AsSpan();
                    firstToken = false;
                    AppendToBytesArray(span, ref bytes, ref bytesIndex);
                }
            }

            string result = Encoding.UTF8.GetString(bytes, 0, bytesIndex);
            ArrayPool<byte>.Shared.Return(bytes);
            return result;
        }

        private void AppendToBytesArray(ReadOnlySpan<char> text, ref byte[] bytes, ref int bytesIndex)
        {
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                if ((uint)c < ByteToUnicodeEncoding.Instance.Count)
                {
                    if (bytesIndex >= bytes.Length)
                    {
                        Helpers.ArrayPoolGrow<byte>(ref bytes, bytes.Length * 2);
                    }

                    bytes[bytesIndex++] = (byte)ByteToUnicodeEncoding.Instance.UnicodeToByte[c];
                    continue;
                }

                // rare cases
                i += EncodeNonAnsiCodePointToUtf8(text, i, ref bytes, ref bytesIndex);
            }
        }

        private static int EncodeNonAnsiCodePointToUtf8(ReadOnlySpan<char> text, int textIndex, ref byte[] destination, ref int bytesIndex)
        {
            Debug.Assert(!text.IsEmpty);
            Debug.Assert(text[textIndex] >= (ByteToUnicodeEncoding.Instance.Count));

            uint c = (uint)text[textIndex];
            if (c <= 0x7FFu)
            {
                // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                if (bytesIndex + 2 > destination.Length)
                {
                    Helpers.ArrayPoolGrow(ref destination, destination.Length * 2);
                }
                destination[bytesIndex] = (byte)((c + (0b110u << 11)) >> 6);
                destination[bytesIndex + 1] = (byte)((c & 0x3Fu) + 0x80u);
                bytesIndex += 2;
                return 0;
            }

            if (textIndex < text.Length - 1 && char.IsSurrogatePair((char)c, text[textIndex + 1]))
            {
                // Scalar 000uuuuu zzzzyyyy yyxxxxxx -> bytes [ 11110uuu 10uuzzzz 10yyyyyy 10xxxxxx ]
                if (bytesIndex + 4 > destination.Length)
                {
                    Helpers.ArrayPoolGrow(ref destination, Math.Max(destination.Length, 4) * 2);
                }

                uint value = (uint)char.ConvertToUtf32((char)c, text[textIndex + 1]);
                destination[bytesIndex] = (byte)((value + (0b11110 << 21)) >> 18);
                destination[bytesIndex + 1] = (byte)(((value & (0x3Fu << 12)) >> 12) + 0x80u);
                destination[bytesIndex + 2] = (byte)(((value & (0x3Fu << 6)) >> 6) + 0x80u);
                destination[bytesIndex + 3] = (byte)((value & 0x3Fu) + 0x80u);
                bytesIndex += 4;
                return 3;
            }

            if (bytesIndex + 3 > destination.Length)
            {
                Helpers.ArrayPoolGrow(ref destination, Math.Max(destination.Length, 3) * 2);
            }

            // Scalar zzzzyyyy yyxxxxxx -> bytes [ 1110zzzz 10yyyyyy 10xxxxxx ]
            destination[bytesIndex] = (byte)((c + (0b1110 << 16)) >> 12);
            destination[bytesIndex + 1] = (byte)(((c & (0x3Fu << 6)) >> 6) + 0x80u);
            destination[bytesIndex + 2] = (byte)((c & 0x3Fu) + 0x80u);
            bytesIndex += 3;
            return 2;
        }

        //
        // Private & Internal methods
        //

        private static void AppendTokenWithOffsetAdjusting(IReadOnlyList<Token> tokensToAdd, List<Token> tokens, int offset, bool addPrefixSpace)
        {
            if (addPrefixSpace)
            {
                if (tokensToAdd.Count > 0)
                {
                    tokens.Add(new Token(tokensToAdd[0].Id, tokensToAdd[0].Value, (offset == 0 ? tokensToAdd[0].Offset.Index : tokensToAdd[0].Offset.Index + offset - 1, offset == 0 ? tokensToAdd[0].Offset.Length - 1 : tokensToAdd[0].Offset.Length)));

                    for (int i = 1; i < tokensToAdd.Count; i++)
                    {
                        tokens.Add(new Token(tokensToAdd[i].Id, tokensToAdd[i].Value, (tokensToAdd[i].Offset.Index + offset - 1, tokensToAdd[i].Offset.Length)));
                    }
                }
            }
            else
            {
                foreach (Token t in tokensToAdd)
                {
                    tokens.Add(new Token(t.Id, t.Value, (t.Offset.Index + offset, t.Offset.Length)));
                }
            }
        }

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private List<Token> EncodeToTokens(Span<char> text, Span<int> mapping, ReadOnlySpan<char> originalText, PriorityQueue<SymbolPair> agenda)
        {
            if (text.Length == 0)
            {
                return [];
            }

            if (text.Length == 1)
            {
                string tokenValue = ByteToUnicodeEncoding.Instance.CharToString[text[0]];
                return new List<Token> { new Token(_vocab[new StringSpanOrdinalKey(tokenValue)].Id, tokenValue, (mapping[0], 1)) };
            }

            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            for (int i = 0; i < text.Length; i++)
            {
                symbols[i] = new BpeSymbol(
                                    prev: i == 0 ? -1 : i - 1,
                                    next: i == text.Length - 1 ? -1 : i + 1,
                                    pieceSpan: (i, 1));

            }

            agenda.Clear();
            for (int i = 1; i < text.Length; i++)
            {
                TryMerge(i - 1, i, text);
            }

            while (agenda.Count > 0)
            {
                SymbolPair top = agenda.Dequeue();

                if (symbols[top.Left].pieceSpan.Length == 0 || symbols[top.Right].pieceSpan.Length == 0 ||
                    symbols[top.Left].pieceSpan.Length + symbols[top.Right].pieceSpan.Length != top.Length)
                {
                    continue;
                }

                // Replaces symbols with `top` rule.
                symbols[top.Left].pieceSpan = (symbols[top.Left].pieceSpan.Index, symbols[top.Left].pieceSpan.Length + symbols[top.Right].pieceSpan.Length);

                // Updates prev/next pointers.
                symbols[top.Left].next = symbols[top.Right].next;

                if (symbols[top.Right].next >= 0)
                {
                    symbols[symbols[top.Right].next].prev = top.Left;
                }
                symbols[top.Right].pieceSpan = (0, 0);

                // Adds new symbol pairs which are newly added after symbol replacement.
                TryMerge(symbols[top.Left].prev, top.Left, text);
                TryMerge(top.Left, symbols[top.Left].next, text);
            }

            List<Token> result = new List<Token>(text.Length);

            for (int index = 0; (uint)index < (uint)text.Length; index = symbols[index].next)
            {
                if (_vocab.TryGetValue(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), out (int Id, string Token) value))
                {
                    result.Add(GetToken(value.Id, value.Token, symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length, originalText, mapping));
                }
                else if (UnknownTokenId.HasValue)
                {
                    result.Add(GetToken(UnknownTokenId.Value, UnknownToken!, symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length, originalText, mapping));
                }
            }

            ArrayPool<BpeSymbol>.Shared.Return(symbols);

            return result;

            static Token GetToken(int id, string token, int index, int length, ReadOnlySpan<char> originalText, Span<int> mapping)
            {
                int tokenStartIndex = mapping[index];
                int tokenLength = (index + length < mapping.Length ? mapping[index + length] - tokenStartIndex : originalText.Length - tokenStartIndex);
                return new Token(id, token, (tokenStartIndex, tokenLength));
            }

            void TryMerge(int left, int right, ReadOnlySpan<char> textSpan)
            {
                if (left == -1 || right == -1)
                {
                    return;
                }

                if (!_mergeRanks.TryGetValue(textSpan.Slice(symbols[left].pieceSpan.Index, symbols[left].pieceSpan.Length), textSpan.Slice(symbols[right].pieceSpan.Index, symbols[right].pieceSpan.Length), out int rank))
                {
                    return;
                }

                SymbolPair pair = new(left, right, rank, symbols[left].pieceSpan.Length + symbols[right].pieceSpan.Length);
                agenda.Enqueue(pair);
            }
        }

        // Added Tokens from https://huggingface.co/Salesforce/codegen-350M-mono/raw/main/tokenizer.json
        internal static readonly Dictionary<string, int> CodeGenAddedTokens = new()
        {
            { "<|endoftext|>",                      50256 },
            { "                               ",    50257 },
            { "                              ",     50258 },
            { "                             ",      50259 },
            { "                            ",       50260 },
            { "                           ",        50261 },
            { "                          ",         50262 },
            { "                         ",          50263 },
            { "                        ",           50264 },
            { "                       ",            50265 },
            { "                      ",             50266 },
            { "                     ",              50267 },
            { "                    ",               50268 },
            { "                   ",                50269 },
            { "                  ",                 50270 },
            { "                 ",                  50271 },
            { "                ",                   50272 },
            { "               ",                    50273 },
            { "              ",                     50274 },
            { "             ",                      50275 },
            { "            ",                       50276 },
            { "           ",                        50277 },
            { "          ",                         50278 },
            { "         ",                          50279 },
            { "        ",                           50280 },
            { "       ",                            50281 },
            { "      ",                             50282 },
            { "     ",                              50283 },
            { "    ",                               50284 },
            { "   ",                                50285 },
            { "  ",                                 50286 },
            { "\t\t\t\t\t\t\t\t\t",                 50287 },
            { "\t\t\t\t\t\t\t\t",                   50288 },
            { "\t\t\t\t\t\t\t",                     50289 },
            { "\t\t\t\t\t\t",                       50290 },
            { "\t\t\t\t\t",                         50291 },
            { "\t\t\t\t",                           50292 },
            { "\t\t\t",                             50293 },
            { "\t\t",                               50294 },
        };

        private static Dictionary<StringSpanOrdinalKey, (int, string)> GetVocabulary(Stream vocabularyStream)
        {
            Dictionary<StringSpanOrdinalKey, (int, string)>? vocab;
            try
            {
                JsonSerializerOptions options = new() { Converters = { StringSpanOrdinalKeyCustomConverter.Instance } };
                vocab = JsonSerializer.Deserialize<Dictionary<StringSpanOrdinalKey, (int, string)>>(vocabularyStream, options) as Dictionary<StringSpanOrdinalKey, (int, string)>;
            }
            catch (Exception e)
            {
                throw new ArgumentException($"Problems met when parsing JSON vocabulary object.{Environment.NewLine}Error message: {e.Message}");
            }

            if (vocab is null)
            {
                throw new ArgumentException($"Failed to read the vocabulary file.");
            }

            return vocab;
        }

        internal static Dictionary<StringSpanOrdinalKeyPair, int> GetMergeRanks(Stream mergeStream)
        {
            var mergeRanks = new Dictionary<StringSpanOrdinalKeyPair, int>();
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

                    mergeRanks.Add(new StringSpanOrdinalKeyPair(line.Substring(0, index), line.Substring(index + 1)), rank++);
                }
            }
            catch (Exception e)
            {
                throw new IOException($"Cannot read the file Merge file.{Environment.NewLine}Error message: {e.Message}", e);
            }

            return mergeRanks;
        }

        private struct SymbolPair : IEquatable<SymbolPair>, IComparable<SymbolPair>
        {
            public int Left { get; set; }
            public int Right { get; set; }
            public int Length { get; set; }
            public int Score { get; set; }

            public SymbolPair(int left, int right, int score, int length)
            {
                Left = left;
                Right = right;
                Score = score;
                Length = length;
            }

            public int CompareTo(SymbolPair other)
            {
                if (Score != other.Score)
                {
                    return Score.CompareTo(other.Score);
                }

                return Left.CompareTo(other.Left);
            }

            public override int GetHashCode()
            {
                int hashCode = 23;
                hashCode = (hashCode * 37) + Score.GetHashCode();
                hashCode = (hashCode * 37) + Left.GetHashCode();
                return hashCode;
            }

            public bool Equals(SymbolPair other) => Left == other.Left && Score == other.Score;
        }

        private record struct BpeSymbol(int prev, int next, (int Index, int Length) pieceSpan);
    }
}
