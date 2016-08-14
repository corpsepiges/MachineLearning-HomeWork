function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% X=(n,j+1) theta=(j+1,1),h=(n,1) also, y=(n,1)  
% n:the nums of data  ;  
% j:the nums of features;
h = sigmoid(X * theta); 
% Cost func 
J = (-log(h')*y - log(ones(1, m) - h')*(ones(m, 1) - y)) / m +(lambda/(2*m)) * sum(theta(2:end).^2);

% Gradient 
grad(1) = (X(:, 1)' * (h - y)) /m;

grad(2:end) = (X(:, 2:end)' * (h - y)) /m + (lambda/m) * theta(2:end); 


% =============================================================

end
