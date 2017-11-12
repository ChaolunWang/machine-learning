%this is the matlab code for lab3 question 3
display('Question3 a:');

A=imread('mandrill.png');                                 %load the picture
R = svd(double(A(1:512, 1:512,1)));                       %doing svd for different page of each color(RGB)
G = svd(double(A(1:512, 1:512,2)));
B = svd(double(A(1:512, 1:512,3)));
figure;                                                   %plot that on one figure
plot(R(1:150), 'r');          
hold;
plot(G(1:150), 'g');

plot(B(1:150), 'b');

hold;

display('Question3 b:');                     
[UR, SR, VR]= svd(double(A(1:512, 1:512,1)));
[UG, SG, VG] = svd(double(A(1:512, 1:512,2)));
[UB, SB, VB] = svd(double(A(1:512, 1:512,3)));

figure;

for i=1:5
    D=4*(2^i);                                              %set rank to different value

    RanknAprox=double(zeros(512,512,3));                    %create the matrix which holds the approximation
    for n=1:D
        RanknAprox(1:512, 1:512,1)=RanknAprox(1:512, 1:512,1)+UR(: , n)*VR(: , n)'*SR(n , n);   %calcualte the approximation 
        RanknAprox(1:512, 1:512,2)=RanknAprox(1:512, 1:512,2)+UG(: , n)*VG(: , n)'*SG(n , n);
        RanknAprox(1:512, 1:512,3)=RanknAprox(1:512, 1:512,3)+UB(: , n)*VB(: , n)'*SB(n , n);
    end
    subplot(2,3,i), imshow(uint8(RanknAprox));              %plot each approximation
end


subplot(2,3,6), imshow(A);



display('Question3 d:')                                   %do the same thing for my picture
X=1200;
Y=1600;
A=imread('mypicture.png');                                 %load the picture
R = svd(double(A(1:X, 1:Y,1)));                       %doing svd for different page of each color(RGB)
G = svd(double(A(1:X, 1:Y,2)));
B = svd(double(A(1:X, 1:Y,3)));
figure;                                                   %plot that on one figure
plot(R(1:150), 'r');          
hold;
plot(G(1:150), 'g');

plot(B(1:150), 'b');

hold;

display('Question3 b:');                     
[UR, SR, VR]= svd(double(A(1:X, 1:Y,1)));
[UG, SG, VG] = svd(double(A(1:X, 1:Y,2)));
[UB, SB, VB] = svd(double(A(1:X, 1:Y,3)));

figure;

for i=1:5
    D=4*(2^i);                                              %set rank to different value

    RanknAprox=double(zeros(X,Y,3));                    %create the matrix which holds the approximation
    for n=1:D
        RanknAprox(1:X, 1:Y,1)=RanknAprox(1:X, 1:Y,1)+UR(: , n)*VR(: , n)'*SR(n , n);   %calcualte the approximation 
        RanknAprox(1:X, 1:Y,2)=RanknAprox(1:X, 1:Y,2)+UG(: , n)*VG(: , n)'*SG(n , n);
        RanknAprox(1:X, 1:Y,3)=RanknAprox(1:X, 1:Y,3)+UB(: , n)*VB(: , n)'*SB(n , n);
    end
    subplot(2,3,i), imshow(uint8(RanknAprox));              %plot each approximation
end


subplot(2,3,6), imshow(A);
