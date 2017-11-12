function [ index ] = organize( B )
index=zeros(1,1);
counter=1;
index(1,1)=B(1,1);
kss=index(1,1);
p=-1;
k=1;
p=index(counter,1);
while( p~=kss||counter==1)
   
   found=false;
   for i=1:size(B)
       if p==B(i,1) && i~=k
          pt=B(i,2);
          kt=i;
          found=true;
       end
   end
   if ~found
        for i=1:size(B)
            if p==B(i,2) && i~=k
                 pt=B(i,1);
                 kt=i;
            end
         end       
   end
   p=pt;
    index=[index;p];
    counter=counter+1;
    k=kt;

end




end

