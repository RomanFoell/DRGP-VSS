function [error_out] = l_inf_error(y_f,y,~)

% l inf error
error_out = max(max(abs(y_f - y)));

end

