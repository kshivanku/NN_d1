function W = randInitializeWeights(L_in, L_out)

	W = zeros(L_in + 1, L_out); %defining W
	init_epsilon = 1;
	W = rand(L_in + 1, L_out) * (2 * init_epsilon) - init_epsilon;

end