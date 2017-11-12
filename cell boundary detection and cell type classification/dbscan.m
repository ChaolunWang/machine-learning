function [ Y, k ] = dbscan( X, r, n )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    temp=X;
    for i=1:size(temp)
       temp(i, 3)=-3; 
    end
    for i=1:size(temp)
        count=0;
        for j=1:size(temp)
            if within(temp(j,1), temp(j,2), temp(i,1), temp(i,2), r)
               count=count+1; 
            end
        end
        if count-1>=n;
            temp(i,3)=0;
        else
            temp(i,3)=-2;
        end
    end
   
    for i=1:size(temp)
        if temp(i,3)~=0;
        contain=false;
        for j=1:size(temp)
            if temp(j,3)==0
                if within(temp(j,1), temp(j,2), temp(i,1), temp(i,2), r)
                	contain=true; 
                end
            end
        end
        if contain;
            temp(i,3)=-1;
        end
        end
    end
    
    %%%%%%%%%%%%%%%%part2
    lable=0;
    for i=1:size(temp)
        if temp(i,3)>=0
            if temp(i,3)==0
                lable=lable+1;
                temp(i,3)=lable;
            
            %%%%%%%%%%%%
            tempcount=0;
            tempcount0=-1;
            while tempcount~=tempcount0
            tempcount0=tempcount;    
                
            for k=1:size(temp)
                if temp(k,3)==lable
                    for j=1:size(temp)     
                        if within(temp(j,1), temp(j,2), temp(k,1), temp(k,2), r) && temp(k,3)~=temp(j,3)
                            if temp(j,3)==0 || temp(j,3)==-1 
                                temp(j,3)=lable;
                                tempcount=tempcount+1;
                            end
                        end
                    end
                end
            end
            
            
            end
            %%%%%%%%%%%%%%%%
            end
        end
    end
    k=lable;
    Y=temp;
end

