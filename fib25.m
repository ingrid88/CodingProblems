function[n] = fib25(num)

n = 3;
    while ((((1+sqrt(5))/2)^n - ((1-sqrt(5))/2)^n)/sqrt(5)) < 10^(num-1)
        n = n+1;
    end
end

% 
% f1 = 1;
% f2 = 1;
% n = 2;
% while f2/10^(num-1) <1
%     n = n + 1;
%     hold = f2;
%     f2 = f1 + f2;
%     f1 = hold;
% end
% end
clc
clear all
close all
format long

tic
phi = (1+sqrt(5))/2;
digits = 1000;
f_term = ceil((digits-1 + log10(5)/2) / log10(phi));
toc
fprintf('The term to contain 1000 digits is %.0d\n',f_term)