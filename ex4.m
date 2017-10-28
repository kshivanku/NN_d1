
clear; close all; clc

%IMPORT THE DATA
load('ex4data1.mat');
m = size(X, 1);

%NEURAL NETWORK STRUCTURE
input_layer = size(X, 2);
hidden_layer = 25;
num_labels = 10;

%INITIALIZE WEIGHTS
initial_theta1 = randInitializeWeights(input_layer, hidden_layer);
initial_theta2 = randInitializeWeights(hidden_layer, num_labels);

initial_nn_params = [initial_theta1(:) ; initial_theta2(:)];

%A FUNCTION THAT GIVES THE VALUE OF COST AND GRADIENT
%separate file

%GRADIENT CHECKING
checkNNGradients;

%TRAIN THE NETWORK
lambda = 1;
costfunction = @(p) nnCostFunction(p, input_layer, hidden_layer, num_labels, X, y, lambda);
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costfunction, initial_nn_params, options);

%CALCULATE EFFECIENCY OF LEARNING
Theta1 = reshape(nn_params(1 : (input_layer+1)*hidden_layer), (input_layer+1), hidden_layer);
Theta2 = reshape(nn_params((1 + ((input_layer+1)*hidden_layer)) : end), (hidden_layer + 1), num_labels);
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);