using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetwork
    {
        private NeuralNetworkConfig _config;
        private NeuralNetworkEngine _engine;

        public double[,] Theta { get; set; }
        public double Lambda { get; set; }

        public NeuralNetwork(NeuralNetworkConfig config)
        {
            if (config == null) throw new ArgumentNullException("config");
            _config = config;
            _engine = new NeuralNetworkEngine(config);
        }

        public NeuralNetworkTrainingResult Train(NeuralNetworkTrainingContext context) 
        {
            var theta = _engine.InitializeTheta().ToArray();
            
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = context.MaxIterations;
            
            alglib.mincgstate state;
            alglib.mincgreport rep;
            alglib.mincgcreate(theta, out state);
            alglib.mincgsetcond(state, epsg, epsf, epsx, maxits);
            alglib.mincgoptimize(state, _engine.CostForAlglib, null, context);
            alglib.mincgresults(state, out theta, out rep);

            //System.Console.WriteLine("{0}", rep.terminationtype);
            //System.Console.WriteLine("{0}", alglib.ap.format(theta, 2));


            return new NeuralNetworkTrainingResult 
            { 
                Theta = null
            };
        }
        public NeuralNetworkTrainingResult TrainWithValidation(NeuralNetworkTrainingContext request)
        {
            throw new NotImplementedException();
        }

        public NeuralNetworkPredictionResult Predict(NeuralNetworkPredictionContext context)
        {
            throw new NotImplementedException();
        }
    }
}
