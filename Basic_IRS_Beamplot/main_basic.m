clc
clear
close all
%% Parameter setting
global c fc d M t1 r1 thetaDeg stepsize
M= 64;%Element number ULA
stepsize=0.1;


fc = 30e9;%Working frequency
c = 3e8;%Light speed
d=(c/fc)/2;%Use half wavelength as distance of element 
thetaDeg=0:stepsize:180;%The discretization of scanning angle
theta = thetaDeg*pi/180;%into arc


%% the Tx position(incident)
t1=90; %The indicent angle towards IRS

%% Rx-pos

r1=130; % It will be used when designing the weights


%% Steering vector calculation 

a1=getAddSteerMatrix(theta,t1*pi/180);
a11=getAddSteerMatrix(r1*pi/180,t1*pi/180);

%% Weights to deploy on IRS

W_wall=ones(M,1); %This is a defalt weights, meaning all weights equal to one, which can be regardes as a wall, thus, only spectrual reflection happen.

W=a11;% Combining as weights take conjuate of channel



%% Quantized

B=1;
W_Q=quanti_bit(B,W);
W_Q=exp(1j*W_Q);


%% Wopt and Beampattern plot


p1=BeampatternPlot(W_Q,a1);
NameArray = {'LineStyle'};
ValueArray = {'-.'}';
set(p1,NameArray,ValueArray)




