function [w ] = weight( A, n )
%calculate the weight of the matrix
    c=(n-1)/2+1;
    w=0;
    for i=1:n
        for j=1:n
            w=w+sqrt(abs(j-c)*abs(i-c))*(A(i,j))^2;
        end
    end
end

