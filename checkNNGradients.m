function checkNNGradients(lambda)

%THINK OF A RANDOM NN STRUCTURE
input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;

%GENERATE RANDOM DATA
m = 5;
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

%GENERATE RANDOM INITIAL WEIGHTS
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
nn_params = [Theta1(:) ; Theta2(:)];

% GET THE VALUE OF COST AND GRAD FROM THE COST FUNCTION
lambda = 3;
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);
 
[cost, grad] = costFunc(nn_params);

%GET THE VALUE OF GRAD FROM METHOD2
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end
