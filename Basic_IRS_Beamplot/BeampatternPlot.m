function [p]=BeampatternPlot(weights,steeringvec)
global thetaDeg 
% offset=1;


Pattern=weights'*steeringvec;

Pattern=abs(Pattern);

normal=Pattern./max(Pattern);
% PatterndB=mag2db(Pattern);
PatterndB=mag2db(normal);


figure();

p=plot(thetaDeg, PatterndB,'LineWidth',3);
grid on;
hold on
legend('Tx1','FontSize', 12)
ylim([-60 0]);
xlim([0,180])
title('Beampattern','FontSize', 20)
xlabel('Angle(Degree)','FontSize', 16)
ylabel( 'Performance(dB)','FontSize', 16)
set(gca,'FontSize',20,'Fontname', 'Times New Roman');

