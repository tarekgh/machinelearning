// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
#pragma warning disable MSML_NoInstanceInitializers
    /// <summary>
    /// The type of the SentencePiece model.
    /// </summary>
    public enum SentencePieceModelType
    {
        /// <summary>
        /// The model type is not defined.
        /// </summary>
        Undefined,

        /// <summary>
        /// The model type is Byte Pair Encoding (Bpe) model.
        /// </summary>
        Bpe,

        /// <summary>
        /// The model type is Unigram model.
        /// </summary>
        Unigram,
    }

    /// <summary>
    /// Options for the SentencePiece tokenizer.
    /// </summary>
    public class SentencePieceOptions
    {
        /// <summary>
        /// The type of the SentencePiece model.
        /// </summary>
        public SentencePieceModelType ModelType { get; set; }

        /// <summary>
        /// Determines whether the model uses a byte fallback strategy to encode unknown tokens as byte sequences.
        /// </summary>
        /// <remarks>
        /// The vocabulary must include a special token for each byte value (0-255) in the format &lt;0xNN&gt;,
        /// where NN represents the byte's hexadecimal value (e.g., &lt;0x41&gt; for byte value 65).
        /// </remarks>
        public bool ByteFallback { get; set; }

        /// <summary>
        /// Indicate emitting the prefix character e.g. U+2581 at the beginning of sentence token during the normalization and encoding.
        /// </summary>
        public bool AddDummyPrefix { get; set; }

        /// <summary>
        /// Indicate if the spaces should be replaced with character U+2581 during the normalization and encoding.
        /// </summary>
        public bool EscapeWhiteSpaces { get; set; } = true;

        /// <summary>
        /// Indicate emitting the character U+2581 at the end of the last sentence token instead beginning of sentence token during the normalization and encoding.
        /// </summary>
        public bool TreatWhitespaceAsSuffix { get; set; }

        /// <summary>
        /// Indicate removing extra white spaces from the original string during the normalization.
        /// </summary>
        public bool RemoveExtraWhiteSpaces { get; set; }

        /// <summary>
        /// Indicate emitting the beginning of sentence token during the encoding.
        /// </summary>
        public bool AddBeginningOfSentence { get; set; } = true;

        /// <summary>
        /// Indicate emitting the end of sentence token during the encoding.
        /// </summary>
        public bool AddEndOfSentence { get; set; }

        /// <summary>
        /// The beginning of sentence token.
        /// </summary>
        public string BeginningOfSentenceToken { get; set; } = "<s>";

        /// <summary>
        /// The end of sentence token.
        /// </summary>
        public string EndOfSentenceToken { get; set; } = "</s>";

        /// <summary>
        /// The unknown token.
        /// </summary>
        public string UnknownToken { get; set; } = "<unk>";

        /// <summary>
        /// The data used for string normalization.
        /// </summary>
        public byte[]? PrecompiledNormalizationData { get; set; }

        /// <summary>
        /// Represent the vocabulary.
        /// The list should be sorted by the token id. Every entry represents a token and its score.
        /// </summary>
        public IEnumerable<KeyValuePair<string, float>>? Vocabulary { get; set; }

        /// <summary>
        /// The special tokens.
        /// Special tokens remain intact during encoding and are not split into sub-tokens.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; set; }
    }
#pragma warning restore MSML_NoInstanceInitializers
}