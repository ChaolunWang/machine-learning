function [ istrue ] = intersect( e1, e2, X )
    istrue=false;
    x1=X(e1(1,1),1);
    x2=X(e1(1,2),1);
    x3=X(e2(1,1),1);
    x4=X(e2(1,2),1);
    y1=X(e1(1,1),2);
    y2=X(e1(1,2),2);
    y3=X(e2(1,1),2);
    y4=X(e2(1,2),2);
   % if (x1 == x2) 
     %   istrue=not(x3 == x4 && x1 ~= x3);
   % elseif (x3 == x4)
    %    istrue=true;
    %else 
  %      m1 = (y1-y2)/(x1-x2);
    %    m2 = (y3-y4)/(x3-x4);
    %    istrue= m1 ~= m2;
   % end
  if ~isempty(polyxpoly([x1;x2],[y1;y2],[x3;x4],[y3;y4]))
     istrue=true; 
  end
    if ((x1 == x3)&&(y1==y3)) || ((x2 == x3)&&(y2==y3)) || ((x1 == x4)&&(y1==y4)) || ((x2 == x4)&&(y2==y4))
        istrue=false;
    end

end

