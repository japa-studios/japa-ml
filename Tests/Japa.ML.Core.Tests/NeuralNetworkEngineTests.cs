using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Japa.ML.Core.Supervised;
using FluentAssertions;

namespace Japa.ML.Core.Tests
{
    [TestClass]
    public class NeuralNetworkEngineTests
    {
        [TestMethod]
        public void InitializeTheta()
        {
            var engine = new NeuralNetworkEngine();
            var theta = engine.InitializeTheta(10, 10);
            theta.Length.Should().Be(100);
            theta.All(i => i > 0 && i < 1);
            theta = engine.InitializeTheta(10, 20);
            theta.Length.Should().Be(200);
            theta.All(i => i > 0 && i < 1);
        }
    }
}
