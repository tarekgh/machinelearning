// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    internal static partial class Helpers
    {
        internal static void ArrayPoolGrow<T>(ref T[] arrayPoolArray, int requiredCapacity)
        {
            T[] tmp = ArrayPool<T>.Shared.Rent(Math.Max(arrayPoolArray.Length * 2, requiredCapacity));
            arrayPoolArray.CopyTo(tmp.AsSpan());
            ArrayPool<T>.Shared.Return(arrayPoolArray);
            arrayPoolArray = tmp;
        }

        public static Dictionary<StringSpanOrdinalKey, int> GetVocabulary(Stream vocabularyStream)
        {
            Dictionary<StringSpanOrdinalKey, int>? vocab;
            try
            {
                JsonSerializerOptions options = new() { Converters = { StringSpanOrdinalKeyConverter.Instance } };
                vocab = JsonSerializer.Deserialize<Dictionary<StringSpanOrdinalKey, int>>(vocabularyStream, options) as Dictionary<StringSpanOrdinalKey, int>;
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

        public static Dictionary<string, int> GetVocab(Dictionary<StringSpanOrdinalKey, int> vocabWithSpanKeys, ref Dictionary<string, int>? vocabOriginal)
        {
            Dictionary<string, int>? publicVocab = Volatile.Read(ref vocabOriginal);
            if (publicVocab is null)
            {
                var vocab = new Dictionary<string, int>();
                foreach (var item in vocabWithSpanKeys)
                {
                    vocab.Add(item.Key.ToString(), item.Value);
                }

                Interlocked.CompareExchange(ref vocabOriginal, vocab, null);
                publicVocab = vocabOriginal;
            }

            return publicVocab;
        }

        public static int EncodeCodePointToUtf8(ReadOnlySpan<char> text, Span<byte> destination)
        {
            Debug.Assert(!text.IsEmpty);
            Debug.Assert(destination.Length >= 4);

            uint c = (uint)text[0];

            if (c <= 0x7Fu)
            {
                destination[0] = (byte)c;
                return 1;
            }

            if (c <= 0x7FFu)
            {
                // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                destination[0] = (byte)((c + (0b110u << 11)) >> 6);
                destination[1] = (byte)((c & 0x3Fu) + 0x80u);
                return 2;
            }

            if (text.Length > 1 && char.IsSurrogatePair((char)c, text[1]))
            {
                // Scalar 000uuuuu zzzzyyyy yyxxxxxx -> bytes [ 11110uuu 10uuzzzz 10yyyyyy 10xxxxxx ]
                uint value = (uint)char.ConvertToUtf32((char)c, text[1]);
                destination[0] = (byte)((value + (0b11110 << 21)) >> 18);
                destination[1] = (byte)(((value & (0x3Fu << 12)) >> 12) + 0x80u);
                destination[2] = (byte)(((value & (0x3Fu << 6)) >> 6) + 0x80u);
                destination[3] = (byte)((value & 0x3Fu) + 0x80u);
                return 4;
            }

            // Scalar zzzzyyyy yyxxxxxx -> bytes [ 1110zzzz 10yyyyyy 10xxxxxx ]
            destination[0] = (byte)((c + (0b1110 << 16)) >> 12);
            destination[1] = (byte)(((c & (0x3Fu << 6)) >> 6) + 0x80u);
            destination[2] = (byte)((c & 0x3Fu) + 0x80u);
            return 3;
        }
    }
}
