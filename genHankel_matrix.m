
function g = genHankel_matrix(A,n)

n1 = size(A,1);
n2 = size(A,2);

g = zeros(n1*(n2-n+1),n);

for x = 1 : n2-n+1
    g( ((x-1)*n1+1) : x*n1, :) = A(:, x:(x+n-1) );
end

end