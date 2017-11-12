function [ alpharad ] = anglevec( veca, vecb )
% Calculate angle between two vectors
alpharad = acos(dot(veca, vecb) / sqrt( dot(veca, veca) * dot(vecb, vecb)))/(2*pi/360);
end