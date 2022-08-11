clear
load('./comsol_data/optimal9.mat')
load parameters.mat

data = readtable('./comsol_data/speaker_9_out.txt');
data = table2array(data);

bp = readtable('./comsol_data/speaker_9_line.txt');
bp = table2array(bp);
bpx = bp(:, 1:2);
bpangle = unwrap(acos(bpx(:, 1)./1000));
a1 = linspace(pi, 0, 181);

bp = bp(:, 3:end);
bpnew = zeros(181, 181);
for ti = 1 : 181
    bpnew(:, ti) = interp1(bpangle, bp(:, ti), a1);
end
bp = bpnew;
% data = table2num(data);

out_comsol = get_lens_out_comsol(data);
out_comsol = out_comsol.';
% out_comsol = out_comsol ./ abs(out_comsol);

len_theta = [len_theta; flipud(len_theta)];
out_matlab = speaker_w * G * diag(len_theta);
beampattern = out_matlab * steerVec;
beampattern2 = out_comsol * steerVec;

max_out = max(abs(out_comsol(:)));
figure(2)
clf
subplot(221)
imagesc((abs(out_comsol)))
colorbar
pbaspect([1,1,1])
subplot(222)
imagesc(abs(out_matlab)')
colorbar
pbaspect([1,1,1])
subplot(223)
plot(sum(abs(out_comsol), 2), 'linewidth', 2)

subplot(224)
plot(sum(abs(out_matlab), 2), 'linewidth', 2)
saveas(gcf, './figs/matlab_test2.png')


figure(1)
clf
subplot(221)
imagesc((abs(beampattern)))
title("MATLAB: lens\_out * steerVec")
pbaspect([1,1,1])
subplot(222)
imagesc((abs(beampattern2)))
title("MATLAB: comsol\_out * steerVec")
pbaspect([1,1,1])

% subplot(223)
% plot(bpangle, 'linewidth', 2)

subplot(224)
% amps = abs(diag(beampattern));
% plot(amps / max(amps), 'linewidth', 3)
% x = 0:181;
% y = linspace(0, 181, 1000);
imagesc(abs(bp)')
title("COMSOL: Simulated beam pattern")
pbaspect([1,1,1])
% subplot(224)
% amps = abs(diag(beampattern2));
% plot(amps / max(amps), 'linewidth', 3)
% pbaspect([1,1,1])
saveas(gcf, 'figs/matlab_test.png')
