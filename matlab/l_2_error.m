function [error_out] = l_2_error(y_f,y,n)

% l 1 error
error_out = mean(sqrt(sum((y_f - y).^2,2)/n));
end
