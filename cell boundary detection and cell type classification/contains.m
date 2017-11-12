function [ ifcontain ] = contains( k0,p )

ifcontain=false;
for i=1:size(k0)
   if(k0(i,1)==p)
       ifcontain=true;
   end
end

end

