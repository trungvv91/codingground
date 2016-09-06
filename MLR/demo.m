clear; clc;

%% Load data
% data = load('ex2data1.txt');
% m = size(data, 1);
% m_train = round(0.8 * m);
% X = data(1:m_train, 1:2); y = data(1:m_train, 3);
% X_test = data(m_train+1:end, 1:2); y_test = data(m_train+1:end, 3);

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);
C = 10;

%% Display data
% figure; hold on;
% pos = find(y == 1); neg = find(y == 0);
% plot(X(pos,1), X(pos,2), 'k+', 'LineWidth', 2);
% plot(X(neg,1), X(neg,2), 'ko', 'MarkerFaceColor', 'y');
% xlabel('x1'); ylabel('x2');
% legend('pos', 'neg');
% hold off;

%% Visualizing f
% W1_vals = linspace(-10, 10);
% W2_vals = linspace(-10, 10);

%% Initialize
n = size(X,1);
X = [ones(n, 1) X];
d = size(X,2);
VW0 = zeros(C*d, 1);

%% MLE
num_iters = 10;
f_history = zeros(num_iters, 1);
lambda = 0;

options = optimset('GradObj', 'on', 'MaxIter', num_iters);
[VW f] = fminunc(@(vw)(MLE(vw, X, y, lambda, C)), VW0, options);

% alpha = 0.5; beta = 0.5;
% for k = 1 : num_iters
%     [f, df] = MLE(W, X, y, lambda);
%     eta = 1;
%     new_f = MLE(W-eta*df, X, y, lambda);
%     while new_f > f - alpha * eta * df' * df
%         eta = eta * beta;
%         new_f = MLE(W-eta*df, X, y, lambda);
%     end
%     W = W - eta * df;
%     f_history(k) = f;
% end
% f_history;

%% Test

pred = zeros(n, C);
for i = 1 : n
    for c = 1 : C
       pred(i, c) = logisticFunc(W, X(i,:)', c, C);
    end
    [t, p] = max(pred(i,:), [], 2);
end

