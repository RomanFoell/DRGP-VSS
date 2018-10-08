function [dquad] = d_eucl(X)

% euclidean distance squared

nX = size(X,2);
% less stable, but more fast
dquad = -2*(X')*X + repmat(sum(X.*X,1)',1,nX) + repmat(sum(X.*X,1),nX,1);
% less fast, but more stable
if sum(any(dquad<0)) > 0 
    for i = 1:nX
        dquad(i,:) = sum((repmat(X(:,i),1,nX) - X).^2,1);
    end
end

end