addpath('./functions/')
% clc
close all

% freq = 10e3:500:30e3;
freq = 20e3;
sweep_angle = [-60:5:60];
amps = zeros(size(freq));
cmap = jet;
index = round(linspace(1, size(cmap, 1), length(sweep_angle)));
cmap = cmap(index, :);

parameter_source = "optimal";
% parameter_source = "raw";

load optimal.mat
load parameters.mat
load qdphase.mat

speaker_w = speaker_w ./ sqrt(sum(speaker_w .* conj(speaker_w), 2));

lens_in = speaker_w * G;
lens_out = lens_in * diag(len_theta);

% lens_out = speaker_w * G * diag(len_theta);
% keyboard


beampattern = lens_out * steerVec;

theta = -90:1:90;
% beampattern = lens_out * steerVec;
figure(2)
set(gcf, 'Position', [123.4000 45.8000 1264 944])
clf
imagesc(abs(beampattern))
% for ti = 1:16
%     subplot(4, 4, ti)
%     imagesc(abs(beampattern(:, :, ti)))
%     title(ti)
%     pbaspect([1,1,1])
% end

saveas(gcf, 'figs/show_results_2D.png')

% keyboard
% w1 = zeros(1, 9);
% w1(5) = 1;
% one_speaker_out = w1 * G * diag(qd) * steerVec;

figure(1)
set(gcf, 'Position', [123.4000 45.8000 1264 944])
clf
subplot(221)
imagesc(theta, theta, abs(beampattern))
colorbar
xlabel('Angles')
ylabel('Code Type')
title('Beam patterns')
set(gca, 'fontsize', 20)
pbaspect([1, 1, 1])

subplot(222)
plot(theta, abs(beampattern(1:10:end, :))', 'linewidth', 2)
xlabel('Angles')
ylabel('Amps')
title('Beam patterns')
set(gca, 'fontsize', 20)
% hold on
% plot(theta, abs(one_speaker_out), 'r', 'linewidth', 5)
pbaspect([1, 1, 1])

subplot(223)
imagesc(unwrap(angle(speaker_w), 2))
colorbar
xlabel('Speaker Index')
ylabel('Direction')
title('Phase of PA Coeffecient')
set(gca, 'fontsize', 20)
pbaspect([1, 1, 1])

subplot(224)
amps = abs(diag(beampattern));
amps = amps / max(amps);
plot(db(amps), 'linewidth', 2)
pbaspect([1, 1, 1])
xlabel('Angles')
ylabel('dB')
title('Signal Strength VS Angles')
set(gca, 'fontsize', 20)
saveas(gcf, 'figs/matlab_test.png')

lens_angle = angle(len_theta);
lens_index = get_lens_index(len_theta);
save('optimal.mat', 'lens_angle', '-append')
save('optimal.mat', 'lens_index', '-append')

figure(2)
clf
plot(lens_angle, '-*', 'linewidth', 2)
hold on
plot(lens_index * 2 * pi / 16, '*')
saveas(gcf, 'figs/matlab_lens_index.png')

% fig = figure(3);

%%

% figure(5)
% clf
% plot(unwrap(angle(lens.delay)), 'linewidth', 2)
% title("lens phase delay")
% xlabel("Lens index")
% ylabel("Phase delay")
% set(gca, 'fontsize', 20)

% figure(6)
% clf
% imagesc(abs(speaker_w))
% xlabel("Array")
% ylabel("Direction")
% title("Amplitude of w")
% set(gca, 'fontsize', 20)

% locs = zeros(length(field_lens.x), 2);
% locs(:, 1) = field_lens.x;
% [x, y] = getCoordinatesCore(field_lens, locs);
% aa = field_lens.sound_pressure(x, y);
% figure(3)
% clf
% plot(abs(aa))
