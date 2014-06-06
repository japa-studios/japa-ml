using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Japa.ML.Core.Supervised
{
    public class NeuralNetworkEngine
    {
        public double[] SigmoidGradient(double[] z)
        {
            //g = sigmoid(z) .* (1-sigmoid(z));
            throw new NotImplementedException();
        }
        public double[] Sigmoid(double[] z)
        {
            //g = 1.0 ./ (1.0 + exp(-z));
            throw new NotImplementedException();
        }
        public double[,] ForwardPropagation(double[,] Theta, double[,] X)
        {
            //% Forward Propagation
            //a1 = [ones(1, m); X'];

            //z2 = Theta1 * a1; % 25x5000
            //a2 = sigmoid(z2);
            //a2 = [ones(1, m); a2];

            //z3 = Theta2 * a2; % 10x5000
            //a3 = sigmoid(z3);
            throw new NotImplementedException();
        }
        public double[,] BackPropagation(double[,] A, double[,] Theta, double[,] Y)
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
        public double CalculateJ(double[] nnOut, double[] y, double lambda, double[,] theta)
        {
            //calcular custo
            //J = (1/m) * sum(((-y_matrix' .* log(a3)) - ((1-y_matrix)' .* log(1-a3)))(:)); 
            //J = J + ( (lambda/(2*m)) * ( sum(Theta1(:, 2:end)(:).^2) + sum(Theta2(:, 2:end)(:).^2) ) );
            return 0;
        }

        public double[,] ComputeGradients(double[,] Theta, double[,] Delta, double lambda)
        {
            //calcular gradients
            //Theta2_grad(:,1) = ((1/m) * Delta2(:,1)); % não regularizar bias
            //Theta2_grad(:,2:end) = ((1/m) * Delta2(:,2:end)) + ((lambda/m) * (Theta2(:,2:end)));
            //Theta1_grad(:,1) = ((1/m) * Delta1(:,1)); % não regularizar bias
            //Theta1_grad(:,2:end) = ((1/m) * Delta1(:,2:end)) + ((lambda/m) * (Theta1(:,2:end)));
            throw new NotImplementedException();
        }
        public void Cost(double[] theta, ref double J, double[] grad, double[,] X, double[] y, double lambda)
        {
            //reshape theta to Theta using Config params
            var Theta = new double[,] { };

            var A = ForwardPropagation(Theta, X);

            //% transformando vetor de labels 'y' em matriz binária com 1 na posição correspondente ao label
            //y_matrix = eye(num_labels)(y,:); % 5000x10
            var Y = new double[,] { };
            var Delta = BackPropagation(A, Theta, Y);


            var thetaGrad = ComputeGradients(Theta, Delta, lambda);

            //% Unroll gradients
            //grad = [Theta1_grad(:) ; Theta2_grad(:)];


        }

        public void Cost(double[] theta, ref double J, double[] grad, object obj)
        {
            var request = obj as NeuralNetworkTrainRequest;
            Cost(theta, ref J, grad, request.X, request.y, request.Lambda);
        }
        public double[] InitializeTheta(uint inputUnits, uint outputUnits)
        {
            double[,] result;
            alglib.ablas.rmatrixmv

            //var W = new double[outputUnits, inputUnits + 1];

            //W = zeros(L_out, 1 + L_in);
            //epsilon_init = 0.12;
            //W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

            throw new NotImplementedException();
        }
    }
}
