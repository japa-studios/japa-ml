using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetworkEngine
    {
        private NeuralNetworkConfig _config;
        public NeuralNetworkEngine(NeuralNetworkConfig config)
        {
            if (config == null) throw new ArgumentNullException("config");
            _config = config;
        }

        public Vector SigmoidGradient(Vector z)
        {
            //g = sigmoid(z) .* (1-sigmoid(z));
            var s = Sigmoid(z);
            Parallel.For(0, z.Count, 
                i => s[i] = s[i] * (1 - s[i]));
            return s;
        }
        public Vector Sigmoid(Vector z)
        {
            //g = 1.0 ./ (1.0 + exp(-z));

            return DenseVector.Create(z.Count, i => 1 / (1 + Math.Pow(Math.E, -1 * z[i])));
        }

        public Matrix[] ForwardPropagation(Matrix[] Theta, Matrix X)
        {
            var m = X.RowCount;
            //% Forward Propagation
            //a1 = [ones(1, m); X'];
            var A1 = X.Transpose().InsertColumn(0, DenseVector.Create(m, i => 1));

            var Z2 = Theta[0] * A1;
            var A2 = Sigmoid(DenseVector.OfVector(Z2.Column(0)));

            //z2 = Theta1 * a1; % 25x5000
            //a2 = sigmoid(z2);
            //a2 = [ones(1, m); a2];

            //z3 = Theta2 * a2; % 10x5000
            //a3 = sigmoid(z3);
            throw new NotImplementedException();
        }

        public Matrix[] BackPropagation(Matrix[] A, Matrix[] Theta, Matrix Y)
        {
            //% Backpropagation
            //d3 = a3' - y_matrix;
            //d3 = d3';

            //d2 = zeros(hidden_layer_size + 1, 1); % 26x1
            //d2 = Theta2' * d3 .* [ones(1,m); sigmoidGradient(z2)];
            //d2 = d2(2:end, :);

            //Delta2 = d3 * a2';
            //Delta1 = d2 * a1';
            throw new NotImplementedException();
        }
        
        public double CalculateJ(Vector nnOut, Vector y, double lambda, Matrix[] Theta)
        {
            //calcular custo
            //J = (1/m) * sum(((-y_matrix' .* log(a3)) - ((1-y_matrix)' .* log(1-a3)))(:)); 
            //J = J + ( (lambda/(2*m)) * ( sum(Theta1(:, 2:end)(:).^2) + sum(Theta2(:, 2:end)(:).^2) ) );
            return 0;
        }
        
        public Matrix[] ComputeGradients(Matrix[] Theta, Matrix[] Delta, double lambda)
        {
            //calcular gradients
            //Theta2_grad(:,1) = ((1/m) * Delta2(:,1)); % não regularizar bias
            //Theta2_grad(:,2:end) = ((1/m) * Delta2(:,2:end)) + ((lambda/m) * (Theta2(:,2:end)));
            //Theta1_grad(:,1) = ((1/m) * Delta1(:,1)); % não regularizar bias
            //Theta1_grad(:,2:end) = ((1/m) * Delta1(:,2:end)) + ((lambda/m) * (Theta1(:,2:end)));
            throw new NotImplementedException();
        }

        public class NeuralNetworkCostResult 
        {
            public double J { get; set; }
            public Matrix[] Gradients { get; set; }
        }

        public NeuralNetworkCostResult Cost(Matrix[] Theta, Matrix X, Vector y, double lambda)
        {
            var A = ForwardPropagation(Theta, X);

            //% transformando vetor de labels 'y' em matriz binária com 1 na posição correspondente ao label
            //y_matrix = eye(num_labels)(y,:); % 5000x10
            var Y = DenseMatrix.Create(y.Count, _config.OutputUnits, (i, j) => y[i]);
            var Delta = BackPropagation(A, Theta, Y);


            var thetaGrad = ComputeGradients(Theta, Delta, lambda);

            //% Unroll gradients
            //grad = [Theta1_grad(:) ; Theta2_grad(:)];
            return new NeuralNetworkCostResult
            {

            };
        }
        public void CostForAlglib(double[] theta, ref double J, double[] grad, object obj)
        {
            var request = obj as NeuralNetworkTrainingContext;
            var layers = _config.Layers;
            var Theta = ReshapeTheta(theta, layers);

            var result = Cost(Theta, request.X, request.y, request.Lambda);

            J = result.J;
            grad = result.Gradients.SelectMany(m => m.ToColumnWiseArray()).ToArray();
        }

        public double[] ReshapeTheta(Matrix[] Theta)
        {
            return Theta.SelectMany(m => m.ToRowWiseArray()).ToArray();
        }
        public Matrix[] ReshapeTheta(double[] theta, int[] layers)
        {
            //reshape theta to Theta using Config params
            var Theta = new Matrix[layers.Length - 1]; //3
            var skip = 0;
            var take = 0;
            for (int i = 0; i < Theta.Length; i++)
            {
                take = layers[i + 1] * (layers[i] + 1);
                Theta[i] = DenseMatrix.OfColumnMajor(
                    layers[i + 1],
                    layers[i] + 1,
                    theta.Skip(skip).Take(take).ToArray());
                skip += take;
            }

            return Theta;
        }

        public Vector InitializeTheta()
        {
            return InitializeTheta(_config.Layers);
        }
        public Vector InitializeTheta(int[] layerSizes)
        {
            //epsilon_init = 0.12;
            //W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
            var layers = layerSizes;
            var vectorSize = layers.Take(layers.Length - 1).Zip(layers.Skip(1), (v, v2) => (v + 1) * v2).Sum(v => v);
            return DenseVector.Create(vectorSize, i => alglib.math.randomreal() * 2 * _config.InitializationEpsilon - _config.InitializationEpsilon);
        }
    }
}
