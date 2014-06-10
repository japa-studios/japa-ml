using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetworkConfig
    {
        public int N { get; set; }
        public int InputUnits { get; set; }
        public int OutputUnits { get; set; }
        public int[] HiddenLayers { get; set; }
        public int[] Layers
        {
            get
            {
                var list = new List<int>(HiddenLayers);
                list.Insert(0, InputUnits);
                list.Add(OutputUnits);
                return list.ToArray();
            }
        }
        public double InitializationEpsilon { get; set; }

        public NeuralNetworkConfig()
        {
            InputUnits = 1;
            OutputUnits = 1;
            HiddenLayers = new int[] { };
            InitializationEpsilon = 0.12;
        }
    }
}
