clear all; close all; clc

RGB=imread('cheekcells.jpg');

imwrite(RGB,'stemcell.png');
RGB=imread('stemcell.png');
figure;
imshow(RGB);
A=rgb2gray(RGB);

grid=5;
limit=0.5;
%{
B=A(891:899, 891:899);
figure;
imshow(B);

FB = fft2(B);
FB = fftshift(FB); % Center FFT

FB = abs(FB); % Get the magnitude
FB = log(FB+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
%FB = mat2gray(FB); % Use mat2gray to scale the image between 0 and 1
FB = mat2gray(FB); % Use mat2gray to scale the image between 0 and 1
figure;
%imshow(FB,[]); % Display the result
colormap('hot');   % set colormap
 imagesc(FB);        % draw image and scale colormap to values range
 colorbar;          % show color scale
 weight(FB, 9)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=A(1:9, 1:9);
figure;
imshow(B);

FB = fft2(B);
FB = fftshift(FB); % Center FFT

FB = abs(FB); % Get the magnitude
FB = log(FB+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
FB = mat2gray(FB); % Use mat2gray to scale the image between 0 and 1
figure;
imshow(FB,[]); % Display the result
colormap('hot');   % set colormap
 imagesc(FB);        % draw image and scale colormap to values range
 colorbar;          % show color scale
 
  weight(FB, 9)
  %}
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  sizea=size(A);
  m=sizea(1,1)-grid;
  n=sizea(1,2)-grid;
  C=zeros(m,n);
  m
  for i=1:m
      for j=1:n
          B=A(i:i+grid-1, j:j+grid-1);
          FB = fft2(B);
          FB = fftshift(FB); % Center FFT
          FB = abs(FB); % Get the magnitude
          FB = log(FB+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
         % FB = mat2gray(FB); % Use mat2gray to scale the image between 0 and 1
          C(i,j)=weight(FB, grid);
      end
      i
  end
 figure;
 colormap('hot');   % set colormap
 imagesc(C);        % draw image and scale colormap to values range
 colorbar;          % show color scale

 
   % 2.0000    8.0000
