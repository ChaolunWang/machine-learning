function [ tf] = within( x0, y0, x1, y1, r )
    if (x0-x1)^2+(y0-y1)^2<r^2
       tf=true; 
    else
        tf=false;
    end
    

end

