function [steeringvec]=getSteeringvector(Angle)
global d c M fc 
% tou=cos(AngleVec)*d/c;
touadd=cos(Angle)*d/c;
a=zeros(M,length(touadd));
for i=1:M
    a(i,:)=exp(-1i*2*pi*fc*touadd*(i-1));
end
steeringvec=a;