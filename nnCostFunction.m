
function [J, grad] = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, lambda)
	%REGENERATE PARAMETERS
	Theta1 = reshape(nn_params(1 : (input_layer + 1) * hidden_layer), input_layer + 1, hidden_layer);
	Theta2 = reshape(nn_params((((input_layer + 1) * hidden_layer) + 1) : end), hidden_layer + 1, num_labels);
	m = size(X, 1);
	%DEFINE VARIABLES
	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));
	%FEEDFORWARD
	for i = 1 : m,
		a1 = X(i, :);
		a1 = [1 a1];
		z2 = a1 * Theta1;
		a2 = sigmoid(z2);
		a2 = [1 a2];
		z3 = a2 * Theta2;
		a3 = sigmoid(z3);
		h = a3;

		yi = zeros(1, num_labels);
		yi(y(i)) = 1;

		J = J + (-1/m) * (yi * log(h') + (1 - yi)* log(1 - h'));

		delta3 = h - yi;
		delta2 = (delta3 * Theta2') .* a2 .* (1 - a2);
		delta2 = delta2(:, (2:end));

		Theta1_grad = Theta1_grad + a1' * delta2;
		Theta2_grad = Theta2_grad + a2' * delta3;
	end
	%Regularizarion
	Theta1_sq = Theta1 .^ 2;
	Theta1_sqSum = sum(sum(Theta1_sq((2:end),:)));
	Theta2_sq = Theta2 .^ 2;
	Theta2_sqSum = sum(sum(Theta2_sq((2:end), :)));
	J = J + (lambda/(2*m)) * (Theta1_sqSum + Theta2_sqSum);

	Theta1_grad = (1/m)*(Theta1_grad);
	Theta1_grad((2:end), :) = Theta1_grad((2:end), :) + (lambda/m) * (Theta1((2:end), :));
	Theta2_grad = (1/m)*(Theta2_grad);
	Theta2_grad((2:end), :) = Theta2_grad((2:end), :) + (lambda/m) * (Theta2((2:end), :));

	grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
