clear all; close all; clc

RGB=imread('tumor.jpg');

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
  m=floor(sizea(1,1)/grid);
  n=floor(sizea(1,2)/grid);
  C=zeros(m,n);
  for i=1:m
      for j=1:n
          B=A((i-1)*grid+1:i*grid, (j-1)*grid+1:j*grid );
          FB = fft2(B);
          FB = fftshift(FB); % Center FFT
          FB = abs(FB); % Get the magnitude
          FB = log(FB+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
         % FB = mat2gray(FB); % Use mat2gray to scale the image between 0 and 1
          C(i,j)=weight(FB, grid);
      end
  end

 figure;
 colormap('hot');   % set colormap
 imagesc(C);        % draw image and scale colormap to values range
 colorbar;          % show color scale
 %{
  sizea=size(A);
  m=sizea(1,1)-grid;
  n=sizea(1,2)-grid;
  C=zeros(m,n);
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
  end
 figure;
 colormap('hot');   % set colormap
 imagesc(C);        % draw image and scale colormap to values range
 colorbar;          % show color scale
%}
 
 %%%%%%%%%%%%threshhold
l=max(max(C));
threshhold=limit*l;
  for i=1:m
      for j=1:n
        if(C(i,j)<threshhold)
            C(i,j)=0;
        else
            C(i,j)=1;
        end

      end
  end

 figure;
 colormap('hot');   % set colormap
 imagesc(C);        % draw image and scale colormap to values range
 colorbar;  
 
 %%%%%%%%%%%%%%%%%%noize removal
   count=0;

  for i=1:m
      for j=1:n
       if(C(i,j)==1)
           count=count+1;
           X(count,1)=j;
           X(count,2)=m-i+1;
       end

      end
  end
  figure;
%Atemp=X(:,2);
%Atemp=Atemp(end:-1:1); 
%X(:,2)=Atemp;
set(gca,'YDir','reverse');
plot(X(:,1),X(:,2), '*');

%%%%%%%%%%%%%%%%%%%%%%%part 3
%DBSCAN
[Y, kc]=dbscan( X, 5, 9 );
figure;
plot(Y(:,1),Y(:,2), '*');
%interest
interest=2
%%%%%%%
figure;
hold on;
kc
for kss=1:kc
count=0;
plotk=kss;
for i=1:size(Y)
      if Y(i,3)==plotk
          count=count+1;
          kdata(count,1)=Y(i,1);
          kdata(count,2)=Y(i,2);
          if Y(i,3)==interest
               Z(count,1)=Y(i,1);
               Z(count,2)=Y(i,2);
          end
      end
end

plot(kdata(:,1),kdata(:,2), '*');
axis([0,250,0,250]);
clearvars kdata;
end
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = gallery('uniformdata',[10,1],0);
y = gallery('uniformdata',[10,1],1);
DT = delaunayTriangulation(Z(:,1),Z(:,2));
DT;
k = convexHull(DT); 
%{
figure
plot(DT.Points(:,1),DT.Points(:,2), '.','markersize',10);
hold on
plot(DT.Points(k,1),DT.Points(k,2),'r')
%}

%%%%%%%part4
figure;
index=concave(k, Z, 60, 10);
index;
plot(Z(index,1),Z(index,2),'b');
axis([0,250,0,250]);


 %8.0000    9.0000    0.9440
   % 9.0000    1.0000    0.9394
   % 3.0000    2.0000    0.4512
   % 1.0000    3.0000    0.3433
   % 2.0000    8.0000
