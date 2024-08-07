﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class KeyToBinaryVectorEstimatorTest : TestDataPipeBase
    {
        public KeyToBinaryVectorEstimatorTest(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public int A;
            public int B;
            public int C;
        }

        private class TestMeta
        {
            [VectorType(2)]
            public string[] A;
            public string B;
            [VectorType(2)]
            public int[] C;
            public int D;
        }

        [Fact]
        public void KeyToBinaryVectorWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingEstimator.ColumnOptions("TermA", "A"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermB", "B"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermC", "C", addKeyValueAnnotationsAsText:true)
                }).Fit(dataView).Transform(dataView);

            var pipe = ML.Transforms.Conversion.MapKeyToBinaryVector(new[] { new InputOutputColumnPair("CatA", "TermA"), new InputOutputColumnPair("CatC", "TermC") });
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void KeyToBinaryVector()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 0),
                new TextLoader.Column("VectorString", DataKind.String, 1, 4),
            });

            var transformedData = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingEstimator.ColumnOptions("A", "ScalarString"),
                new ValueToKeyMappingEstimator.ColumnOptions("B", "VectorString") })
                .Fit(data).Transform(data);


            var est = ML.Transforms.Conversion.MapKeyToBinaryVector("ScalarString", "A")
                .Append(ML.Transforms.Conversion.MapKeyToBinaryVector("VectorString", "B"));

            TestEstimatorCore(est, transformedData, invalidInput: data);

            Done();
        }

        [Fact]
        public void TestMetadataPropagation()
        {
            var data = new[] {
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 5,3}, D= 1},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6} };


            var dataView = ML.Data.LoadFromEnumerable(data);
            var termEst = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingEstimator.ColumnOptions("TA", "A", addKeyValueAnnotationsAsText: true),
                new ValueToKeyMappingEstimator.ColumnOptions("TB", "B", addKeyValueAnnotationsAsText: true),
                new ValueToKeyMappingEstimator.ColumnOptions("TC", "C"),
                new ValueToKeyMappingEstimator.ColumnOptions("TD", "D") });
            var termTransformer = termEst.Fit(dataView);
            dataView = termTransformer.Transform(dataView);

            var pipe = ML.Transforms.Conversion.MapKeyToBinaryVector(new[] {
                new InputOutputColumnPair("CatA", "TA"),
                new InputOutputColumnPair("CatB", "TB"),
                new InputOutputColumnPair("CatC", "TC"),
                new InputOutputColumnPair("CatD", "TD")
            });

            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "Bit1", "Bit0" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Empty(column.Annotations.Schema);

            column = result.Schema["CatD"];
            Assert.Equal(AnnotationUtils.Kinds.IsNormalized, column.Annotations.Schema.Single().Name);
            Assert.True(column.IsNormalized());
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} xf=KeyToBinary{col=C:B} in=f:\2.txt" }));
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var est = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingEstimator.ColumnOptions("TermA", "A"),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermB", "B", addKeyValueAnnotationsAsText:true),
                    new ValueToKeyMappingEstimator.ColumnOptions("TermC", "C")
            });
            var transformer = est.Fit(dataView);
            dataView = transformer.Transform(dataView);
            var pipe = ML.Transforms.Conversion.MapKeyToBinaryVector(new[] { new InputOutputColumnPair("CatA", "TermA"), new InputOutputColumnPair("CatB", "TermB"), new InputOutputColumnPair("CatC", "TermC") });
            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }
    }
}
