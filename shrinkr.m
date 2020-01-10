function A = shrinkr(B,gamma)
A = sign(B).*max(abs(B)-gamma,0);
A(A~=0&A>0) = A(A~=0&A>0) + gamma;
A(A~=0&A<0) = A(A~=0&A<0) - gamma;
end