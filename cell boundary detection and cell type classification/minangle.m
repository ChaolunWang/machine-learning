function [ index, min ] = minangle( e, X )
index=0;
p1=e(1,1);
p2=e(1,2);
ve1=[X(p2,1)-X(p1,1), X(p2,2)-X(p1,2)];
ve2=[X(p1,1)-X(p2,1), X(p1,2)-X(p2,2)];
min=1000;
for i=1:size(X)
    if (i~=p1)&&(i~=p2)
        vp1=[X(i,1)-X(p1,1), X(i,2)-X(p1,2)];
        vp2=[X(i,1)-X(p2,1), X(i,2)-X(p2,2)];
        temp=max(anglevec(ve1, vp1), anglevec(ve2, vp2));
        if(temp<min)
            min=temp;
            index=i;
        end
    end
end

end

