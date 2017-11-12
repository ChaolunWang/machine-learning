function [ index ] = concave( k, X , TOL, EDGE);

    counter=0;

    for i=1:size(k)-1
       A(i,1)=k(i,1);
       A(i,2)=k(i+1,1);
       counter=counter+1;
    end
    k0=k(1:counter,1);
    for i=1: size(A)
       A(i,3)=sqrt((X(A(i,1), 1) -X(A(i,2), 1))^2+(X(A(i,1), 2)-X(A(i,2), 2))^2 );
    end
    [A_sorted sorted_index] = sort(A);
    A_sorted(:,1) = A(sorted_index(:,3),1);
    A_sorted(:,2) = A(sorted_index(:,3),2);
    A=A_sorted;
    B = zeros(0,3);
    A
    while (counter>0)
    
        e=A(counter,:)      
        A(counter,:)=[];
        counter=counter-1;
        added=false;
        if(e(1,3)>EDGE)
        [p0,minp]=minangle(e, X);
        minp
       
        p0
        k0
        if(minp<TOL)&& ~(contains(k0,p0))
            e1=[p0, e(1,1), sqrt((X(p0, 1) -X(e(1,1), 1))^2+(X(p0, 2)-X(e(1,1), 2))^2 )];
            e2=[p0, e(1,2), sqrt((X(p0, 1) -X(e(1,2), 1))^2+(X(p0, 2)-X(e(1,2), 2))^2 )];
            e1
            e2
            condition=true;
            for i=1:counter
                temp=A(i,:); 
                temp
                if intersect(temp, e1, X)
                    condition=false;
                end
                condition
                 if intersect(temp, e2, X)
                    condition=false;
                 end
                condition
            end
            if(condition)
               A=[e1;e2;A]; 
               A
               counter=counter+2;
               [A_sorted sorted_index] = sort(A);
               A_sorted(:,1) = A(sorted_index(:,3),1);
               A_sorted(:,2) = A(sorted_index(:,3),2);
               A=A_sorted;
               k0=[k0;p0];
               added=true;
            end

        end
        end
       if ~added
            B=[B;e]; 
        end
      
    end
    B
    index=organize(B);
end

