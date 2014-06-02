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

        private double ForwardPropagation(double[] X, double[] y, double lambda, double[,] theta)
        {
            //% Forward Propagation
            //a1 = [ones(1, m); X'];

            //z2 = Theta1 * a1; % 25x5000
            //a2 = sigmoid(z2);
            //a2 = [ones(1, m); a2];

            //z3 = Theta2 * a2; % 10x5000
            //a3 = sigmoid(z3);

            return 1;
        }

        private double CalculateJ(double[] nnOut, double[] y, double lambda, double[,] theta)
        {
            //calcular custo
            //J = (1/m) * sum(((-y_matrix' .* log(a3)) - ((1-y_matrix)' .* log(1-a3)))(:)); 
            //J = J + ( (lambda/(2*m)) * ( sum(Theta1(:, 2:end)(:).^2) + sum(Theta2(:, 2:end)(:).^2) ) );
            return 0;
        }

        private void Cost(double[] x, ref double J, double[] grad, object obj)
        {
            //reshape x to X using Config params
            //reshape grad to Grad using Config params

            //% Forward Propagation
            //a1 = [ones(1, m); X'];

            //z2 = Theta1 * a1; % 25x5000
            //a2 = sigmoid(z2);
            //a2 = [ones(1, m); a2];

            //z3 = Theta2 * a2; % 10x5000
            //a3 = sigmoid(z3);

            //var nnOut = 

            //% transformando vetor de labels 'y' em matriz binária com 1 na posição correspondente ao label
            //y_matrix = eye(num_labels)(y,:); % 5000x10

            //calcular custo
            //J = (1/m) * sum(((-y_matrix' .* log(a3)) - ((1-y_matrix)' .* log(1-a3)))(:)); 
            //J = J + ( (lambda/(2*m)) * ( sum(Theta1(:, 2:end)(:).^2) + sum(Theta2(:, 2:end)(:).^2) ) );
            //J = CalculateJ()

            //% Backpropagation
            //d3 = a3' - y_matrix;
            //d3 = d3';

            //d2 = zeros(hidden_layer_size + 1, 1); % 26x1
            //d2 = Theta2' * d3 .* [ones(1,m); sigmoidGradient(z2)];
            //d2 = d2(2:end, :);

            //Delta2 = d3 * a2';
            //Delta1 = d2 * a1';

            //calcular gradients
            //Theta2_grad(:,1) = ((1/m) * Delta2(:,1)); % não regularizar bias
            //Theta2_grad(:,2:end) = ((1/m) * Delta2(:,2:end)) + ((lambda/m) * (Theta2(:,2:end)));
            //Theta1_grad(:,1) = ((1/m) * Delta1(:,1)); % não regularizar bias
            //Theta1_grad(:,2:end) = ((1/m) * Delta1(:,2:end)) + ((lambda/m) * (Theta1(:,2:end)));

            //% Unroll gradients
            //grad = [Theta1_grad(:) ; Theta2_grad(:)];



        }

        public void function1_grad(double[] x, ref double func, double[] grad, object obj)
        {
            // this callback calculates f(x0,x1) = 100*(x0+3)^4 + (x1-3)^4
            // and its derivatives df/d0 and df/dx1
            func = 100 * System.Math.Pow(x[0] + 3, 4) + System.Math.Pow(x[1] - 3, 4);
            grad[0] = 400 * System.Math.Pow(x[0] + 3, 3);
            grad[1] = 4 * System.Math.Pow(x[1] - 3, 3);
        }
        public NeuralNetworkTrainResult Train(NeuralNetworkTrainRequest r) 
        {
            //Reshape X to a vector
            //Ramdom initialize Theta as a matrix
            double[] Theta = null;

            double[] x = new double[] { 0, 0 };
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = 0;
            alglib.mincgstate state;
            alglib.mincgreport rep;

            alglib.mincgcreate(x, out state);
            alglib.mincgsetcond(state, epsg, epsf, epsx, maxits);
            alglib.mincgoptimize(state, function1_grad, null, null);
            alglib.mincgresults(state, out x, out rep);

            System.Console.WriteLine("{0}", rep.terminationtype); // EXPECTED: 4
            System.Console.WriteLine("{0}", alglib.ap.format(x, 2)); // EXPECTED: [-3,3]

            return new NeuralNetworkTrainResult 
            { 
                Theta = null
            };

            //Reshape Theta as vector
        }
    }

    public class NeuralNetworkTrainRequest
    {
        public double[,] X { get; set; }
        public double[] Y { get; set; }
        public double Lambda { get; set; }
    }

    public class NeuralNetworkTrainResult 
    {
        public double[,] Theta { get; set; }
    }

    public class NeuralNetworkConfig
    {
        public uint InputUnits { get; set; }
        public uint OutputUnits { get; set; }
        public uint[] HiddenLayers { get; set; }
        public uint MaxIteration { get; set; }
        public bool DoNotUseRegularization { get; set; }
        public bool CheckGradient { get; set; }
    }
}
