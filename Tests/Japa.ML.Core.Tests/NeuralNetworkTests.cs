using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Japa.ML.Core.Supervised;

namespace Japa.ML.Core.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void Training()
        {
            var nn = new NeuralNetwork(new NeuralNetworkConfig
            {
                InputUnits = 400,
                HiddenLayers = new uint[] { 25 },
                OutputUnits = 10
            });
            var result = nn.Train(new NeuralNetworkTrainRequest());
        }
        
    }
}
