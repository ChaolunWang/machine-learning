%this is the matlab code for lab 3 question 1
A=[1 2 3; 4 5 6; 7 8 9; 10 11 12];                     %initialize A
[U, S, V]=svd(A);                                      %using SVD to solve A
U
S
V
Rank1Aprox=U(: , 1)*V(: , 1)'*S(1 , 1)                 %calculate rank 1 approximation

RelativeError1=0;                                      %calculate relative error by method 1
Total=0;
for i=1:4
    for j=1:3
        RelativeError1=RelativeError1+(Rank1Aprox(i, j)-A(i, j))^2;
        Total=Total+A(i,j)^2;
    end
end

RelativeError1=sqrt(RelativeError1)/sqrt(Total);
RelativeError1

r=2;
p=1;
term1=0;
term2=0;

for i=(p+1):r                                           %calculate relative error by method 2
    term1=term1+S(i, i)^2;
end

for i=1:r
    term2=term2+S(i, i)^2;
end

RelativeError2=sqrt(term1/term2);
RelativeError2
