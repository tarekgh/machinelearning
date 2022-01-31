﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Globalization;
using System.Threading;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.SearchSpace.Option;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    public class ParameterTest : TestBase
    {
        public ParameterTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void Parameter_with_single_value_test()
        {
            var parameter = Parameter.FromInt(3);

            // integer
            parameter.AsType<int>().Should().Be(3);

            // double
            parameter = Parameter.FromDouble(3.1415926);
            parameter.AsType<double>().Should().Be(3.1415926);

            // float
            parameter = Parameter.FromFloat(3.1415926F);
            parameter.AsType<float>().Should().Be(3.1415926F);

            // string
            parameter = Parameter.FromString("abc");
            parameter.AsType<string>().Should().Be("abc");

            // bool
            parameter = Parameter.FromBool(false);
            parameter.AsType<bool>().Should().BeFalse();

            // string[]
            parameter = Parameter.FromIEnumerable(new[] { "a", "b", "c" });
            parameter.AsType<string[]>().Should().BeEquivalentTo("a", "b", "c");

            // enum
            parameter = Parameter.FromEnum(JTokenType.Array);
            parameter.AsType<JTokenType>().Should().Be(JTokenType.Array);

            // long
            parameter = Parameter.FromLong(long.MaxValue);
            parameter.AsType<long>().Should().Be(long.MaxValue);
        }

        [Fact]
        public void Nested_parameter_test()
        {
            var b = new B();
            var parameter = Parameter.FromObject(b);

            parameter["Int"].AsType<int>().Should().Be(0);
            parameter["Float"].AsType<float>().Should().Be(1f);
            parameter["Double"].AsType<double>().Should().Be(2);
            parameter["Bool"].AsType<bool>().Should().BeFalse();
            parameter["String"].AsType<string>().Should().Be("String");
            parameter["Strings"].AsType<string[]>().Should().BeEquivalentTo("A", "B", "C");
            parameter["JTokenType"].AsType<JTokenType>().Should().Be(JTokenType.Null);
            parameter["A"].AsType<A>().Strings.Should().BeEquivalentTo("A", "B", "C");
            parameter.AsType<B>().Strings.Should().BeEquivalentTo("A", "B", "C");
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void Nested_parameter_serialize_test()
        {
            var b = new B();
            b.String = null;
            var parameter = Parameter.FromObject(b);
            var json = JsonConvert.SerializeObject(parameter);
            Approvals.Verify(json);

            parameter = JsonConvert.DeserializeObject<Parameter>(json);
            parameter["Int"].AsType<int>().Should().Be(0);
            parameter["Float"].AsType<float>().Should().Be(1f);
            parameter["Double"].AsType<double>().Should().Be(2);
            parameter["Bool"].AsType<bool>().Should().BeFalse();
            parameter["Strings"].AsType<string[]>().Should().BeEquivalentTo("A", "B", "C");
            parameter["JTokenType"].AsType<JTokenType>().Should().Be(JTokenType.Null);
            json = JsonConvert.SerializeObject(parameter);
            Approvals.Verify(json);
        }

        [Fact]
        public void Parameter_AsType_should_be_culture_invariant()
        {
            var originalCuture = Thread.CurrentThread.CurrentCulture;
            var culture = new CultureInfo("ru", false);
            Thread.CurrentThread.CurrentCulture = culture;
            var ss = new SearchSpace();
            ss.Add("_SampleSize", new UniformDoubleOption(10000, 20000));
            var parameter = ss.SampleFromFeatureSpace(new[] { 0.5 });
            parameter["_SampleSize"].AsType<double>().Should().Be(15000.0);
            Thread.CurrentThread.CurrentCulture = originalCuture;
        }

        private class A
        {
            public int Int { get; set; } = 0;

            public float Float { get; set; } = 1f;

            public double Double { get; set; } = 2;

            public bool Bool { get; set; } = false;

            public string String { get; set; } = "String";

            public string[] Strings { get; set; } = new[] { "A", "B", "C" };

            public JTokenType JTokenType { get; set; } = JTokenType.Null;
        }

        private class B : A
        {
            public A A { get; set; } = new A();
        }
    }
}
