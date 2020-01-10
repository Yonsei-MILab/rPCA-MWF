function g = pons_matrix(A,is)
% projection on to null space

n1 = size(A,1);
n2 = size(A,2);

np = n1 / is;
nl = (is-1) + n2;

g = zeros(np,nl);
countf = zeros(np,nl);

for x = 1 : is
        g(:,x:x+n2-1) = g(:,x:x+n2-1) + A( ((x-1)*np+1):(x*np) ,:);
        countf(:,x:x+n2-1) = countf(:,x:x+n2-1) + 1;
end
g = g ./ countf;
end