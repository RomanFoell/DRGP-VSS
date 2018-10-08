function [error_out] = l_1_error(y_f,y,n)

% l 1 error
error_out = mean(sum(abs(y_f - y),2)/n);
end

