using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetwork
    {
        public NeuralNetworkConfig Config { get; set; }

        public NeuralNetwork(NeuralNetworkConfig config)
        {
            Config = config;
        }

        public NeuralNetworkTrainResult Train(NeuralNetworkTrainRequest request) 
        {
            //Reshape X to a vector
            //Ramdom initialize Theta as a matrix

            var theta = InitializeTheta(Config.InputUnits, Config.OutputUnits);
            
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = request.MaxIterations;
            
            alglib.mincgstate state;
            alglib.mincgreport rep;
            alglib.mincgcreate(theta, out state);
            alglib.mincgsetcond(state, epsg, epsf, epsx, maxits);
            alglib.mincgoptimize(state, Cost, null, request);
            alglib.mincgresults(state, out theta, out rep);

            System.Console.WriteLine("{0}", rep.terminationtype); // EXPECTED: 4
            System.Console.WriteLine("{0}", alglib.ap.format(theta, 2)); // EXPECTED: [-3,3]

            return new NeuralNetworkTrainResult 
            { 
                Theta = null
            };
        }
        public NeuralNetworkTrainResult TrainWithValidation(NeuralNetworkTrainRequest request)
        {
            throw new NotImplementedException();
        }
    }

    public class NeuralNetworkTrainRequest
    {
        public double[,] X { get; set; }
        public double[] y { get; set; }

        public int MaxIterations { get; set; }

        public bool UseRegularization { get; set; }
        public double Lambda { get; set; }

        public NeuralNetworkTrainRequest()
        {
            UseRegularization = true;
            Lambda = 1;
            MaxIterations = 5000;
        }
    }

    public class NeuralNetworkTrainResult 
    {
        public double[,] Theta { get; set; }
        public NeuralNetworkTrainReport Report { get; set; }
    }
    public class NeuralNetworkTrainReport
    {
        public double[,] LearningCurve { get; set; }
    }
    public class NeuralNetworkConfig
    {
        public uint InputUnits { get; set; }
        public uint OutputUnits { get; set; }
        public uint[] HiddenLayers { get; set; }
        public double InitializationEpsilon { get; set; }

        public NeuralNetworkConfig()
        {
            InputUnits = 1;
            OutputUnits = 1;
            HiddenLayers = new uint[] { };
            InitializationEpsilon = 0.12;
        }
    }
}
