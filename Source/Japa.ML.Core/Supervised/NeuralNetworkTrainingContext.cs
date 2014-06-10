using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetworkTrainingContext
    {
        public Matrix X { get; set; }
        public Vector y { get; set; }

        public int MaxIterations { get; set; }

        public double Lambda { get; set; }

        public NeuralNetworkTrainingContext()
        {
            Lambda = 1;
            MaxIterations = 5000;
        }
    }
}
