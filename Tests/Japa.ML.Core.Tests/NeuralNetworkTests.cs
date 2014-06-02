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

            });
            nn.Train(new NeuralNetworkTrainRequest { });
        }
    }
}
