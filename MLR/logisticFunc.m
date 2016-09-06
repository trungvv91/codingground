function g = logisticFunc(W, x_i, y_i, C)

g = 0;
temp = 0;
for c = 1 : C
    W_c = W(c,:)';
    temp = temp + (c == y_i) * W_c' * x_i;
    g = g + exp(W_c' * x_i);
end

g = exp(temp) / g;

end

