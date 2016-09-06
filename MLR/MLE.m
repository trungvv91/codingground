function [ f, df ] = MLE( VW, X, y, lambda, C )

[n, d] = size(X);
f = 0;
W = reshape(VW, d, C)';
df = zeros(size(W));

for i = 1 : n
    x_i = X(i,:)';
    temp = 0;
    W_y = W(y(i), :)';
    for c = 1 : C
        W_c = W(c, :)';        
        temp = temp + exp((W_c - W_y)' * x_i);
        df(c,:) = df(c,:) + ( logisticFunc(W, x_i, y(i), C) - (y(i)==c) ) * x_i';
    end
    f = f + log(temp);
end

temp = 0;
for c = 1 : C
    W_c = W(c, :)';
    temp = temp + (W_c' * W_c);
end
f = f + 0.5 * lambda * temp;
df(:, 2:end) = df(:, 2:end) + lambda * W(:, 2:end);
df = reshape(df', C*d, 1);

% f = f / n;
% df = df / n;
