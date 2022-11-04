freq = 20e9;
c = physconst('lightspeed');
elementspacing = 5.3/1000;
numElems = 8;
array = phased.URA([numElems,numElems],elementspacing);
% array = phased.ULA(numElems,elementspacing);
azSteer = 0;
focalRange = 20e-3;
SV = phased.FocusedSteeringVector('SensorArray',array,'PropagationSpeed',c);
AR = phased.SphericalWavefrontArrayResponse('SensorArray',array,'PropagationSpeed',c,'WeightsInputPort',true); %,'WeightsInputPort',true
SV_dir = phased.SteeringVector('SensorArray',array);
ang = [0;0];
weights_dirr = SV_dir(freq,ang);
arrayLength = numElems*elementspacing;
x = linspace(1e-3,4*focalRange,200);
arrayLength_total = 31*elementspacing;
y = linspace(-arrayLength_total/2,arrayLength_total/2,200);
[az,el,rng] = cart2sph(x,y',0);
ang = rad2deg([az(:) el(:)]');
rng = rng(:)';
weights = SV(freq,[azSteer;0],focalRange);
phases = rad2deg(angle(weights(:)));
phases_new = 0 - phases;
weights_array = reshape(weights, numElems, numElems);
phase_weights = rad2deg(angle(weights_array(:,:)));
weights_with_dir = weights .* weights_dirr;
beam = AR(freq,ang,rng,weights_with_dir);
beam = reshape(beam,numel(y),numel(x));
beam_value = abs(beam);
beam_nor = beam/max(abs(beam(:)));
beam_nor = mag2db(abs(beam_nor));
% disp("max power: ")
% disp(max(max(beam_value(:, 48:105))))
% A = [1,1,1,1];
% disp("antenna 1-4: ")
% A(1) = mean(max(beam_value(42:63, 48:55)));
% A(2) = mean(max(beam_value(72:93, 48:55)));
% A(3) = mean(max(beam_value(102:123, 48:55)));
% A(4) = mean(max(beam_value(132:153, 48:55)));
% disp(A);
% disp("fixed antenna 1")
% B = mean(max(beam_value(90:110, 48:55)));
% disp(B);



% disp(mean(max(beam_value(48:55, 48:51))))
% disp(mean(max(beam_value(72:79, 48:51))))
% disp(mean(max(beam_value(122:129, 48:51))))
% disp(mean(max(beam_value(146:153, 48:51))))
figure
helperPlotResponse(beam_nor,x,y,array)
title('Single Steered and Focused Beam')

saveas(gcf, 'figs/test.png')

function helperPlotResponse(R,x,y,array)
% Plot the response, R, on the domain defined by x and y.
imagesc(x,y,R)
set(gca,'ydir','normal')
caxis([-32 0])
xlabel('Axial Distance')
ylabel('Lateral Position')
end
% i