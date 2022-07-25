% clear all


% B=3;
% x= 2*pi*rand(8,1);
% quanti_(B,x)


function [new]=quanti_bit(B,x)


Q=2^B ;%length of quanti bits;
% pp=linspace(0,2*pi-2*pi/Q,Q);
pp=linspace(0,2*pi,Q+1);

disp(pp)

% for i=1:length(x) 
%     for j=1:length(pp)-1
%      if abs(x(i)-pp(j+1))<2*pi/Q
%          x(i)=pp(j);
%      end
%         
%         
% end
for i=1:length(x)
    kk=abs(pp-(angle(x(i))+pi));
    [M,I]=min(kk(:));
    x(i)=pp(I)-pi;
    
end
new=x;
    

end

% for i=1:length(x)
%     kk=abs(pp-(angle(x(i))+pi));
%     [M,I]=min(kk(:));
%     x(i)=pp(I)-pi;
%     
% end