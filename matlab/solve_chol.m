function [alpha] = solve_chol(L,y)

alpha = L'\(L\y);

end