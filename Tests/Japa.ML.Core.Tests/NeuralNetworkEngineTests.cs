using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Japa.ML.Core.Supervised;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Japa.ML.Core.Tests
{
    [TestClass]
    public class NeuralNetworkEngineTests
    {
        private NeuralNetworkConfig _config400x25x10;
        private NeuralNetworkConfig _config1x2X2x1;
        public NeuralNetworkEngineTests()
        {
            _config400x25x10 = new NeuralNetworkConfig
            {
                InputUnits = 400,
                HiddenLayers = new int[] { 25 },
                OutputUnits = 10
            };
            _config1x2X2x1 = new NeuralNetworkConfig
            {
                InputUnits = 1,
                HiddenLayers = new int[] { 2, 2 },
                OutputUnits = 1
            };
        }

        [TestMethod]
        public void InitializeTheta()
        {
            var engine = new NeuralNetworkEngine(_config400x25x10);
            var theta = engine.InitializeTheta();
            theta.Count.Should().Be(25*401 + 10*26);
            theta.All(i => i > 0 && i < 1);
        }
        [TestMethod]
        public void Sigmoid()
        {
            var engine = new NeuralNetworkEngine(_config400x25x10);
            
            var sigmoid = engine.Sigmoid(new DenseVector(new double[] { 100, 0, -100 }));
            sigmoid.Should().NotBeNull().
                And.HaveCount(3);
            sigmoid[0].Should().BeApproximately(1, 0.01);
            sigmoid[1].Should().Be(0.5);
            sigmoid[2].Should().BeApproximately(0, 0.01);
        }
        [TestMethod]
        public void SigmoidGradient()
        {
            var engine = new NeuralNetworkEngine(_config400x25x10);
            var sigmoid = engine.SigmoidGradient(new DenseVector(new double[] { 0 }));
            sigmoid.Should().NotBeNull().
                And.HaveCount(1);
            sigmoid[0].Should().Be(0.25);
        }
        [TestMethod]
        public void ReshapeTheta()
        {
            var engine = new NeuralNetworkEngine(_config1x2X2x1);

            var vector = new double[] { 11, 12, 21, 22, 11, 12, 13, 11, 12, 21, 22, 31, 32 };
            var matrices = engine.ReshapeTheta(vector.ToArray(), new int[] { 1, 2, 1, 3 });
            var reshaped = engine.ReshapeTheta(matrices);
            reshaped.Should().BeEquivalentTo(vector.ToArray());

            var theta = engine.InitializeTheta();
            var Theta = engine.ReshapeTheta(theta.ToArray(), _config1x2X2x1.Layers);
            Theta.Length.Should().Be(3);
            Theta[0].RowCount.Should().Be(2);
            Theta[0].ColumnCount.Should().Be(2);
            Theta[1].RowCount.Should().Be(2);
            Theta[1].ColumnCount.Should().Be(3);
            Theta[2].RowCount.Should().Be(1);
            Theta[2].ColumnCount.Should().Be(3);
            var thetaVector = engine.ReshapeTheta(Theta);
            thetaVector.Should().BeEquivalentTo(theta.ToArray());
        }

    }
}
