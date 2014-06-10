using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetworkTrainingResult
    {
        public ICollection<Matrix> Theta { get; set; }
        public NeuralNetworkTrainingReport Report { get; set; }
    }
}
