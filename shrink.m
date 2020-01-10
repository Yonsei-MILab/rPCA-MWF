function A = shrink(B,gamma)
A = sign(B).*max(abs(B)-gamma,0);
end