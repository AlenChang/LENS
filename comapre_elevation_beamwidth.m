% ele_lens = rx1(:, 61);
% ele_array = rx2(:, 61);
% azi_lens = rx1(61, :);
% azi_array = rx2(61, :);
params.c = 34300;
params.fc = 20e3;
params.lambda = params.c / params.fc;
num_speaker = 16;
speaker_locs = generate1Dlocs(params, num_speaker);
speaker_weights = getPhasedArraySteeringWeights(params, num_speaker);

elevation_theta = 0 : 0.1 : 180;
azi_lens = helperPlotBeamNormal(params, lens_locs, lens_out(91-30, :), elevation_theta, 'x-axis');

azi_array = helperPlotBeamNormal(params, speaker_locs, speaker_weights(91, :), elevation_theta, 'x-axis');

ele_lens = helperPlotBeamNormal(params, lens_locs, lens_out(91-30, :), elevation_theta, 'y-axis');

ele_array = helperPlotBeamNormal(params, speaker_locs, speaker_weights(91, :), elevation_theta, 'x-axis');

ele_lens = ele_lens / max(abs(ele_lens));
ele_array = ele_array / max(abs(ele_array));

azi_lens = azi_lens / max(abs(azi_lens));
azi_array = azi_array / max(abs(azi_array));

x = elevation_theta;
figure(1)
colors = colororder;
clf
y1 = db(abs(ele_lens).^2) / 2;
y2 = db(abs(ele_array).^2) / 2;
h1 = plot(x, y2, 'linewidth', 3, 'Color', colors(3,:));
hold on
h2 = plot(xlim, [-3, -3], '--', 'linewidth', 3, 'Color', colors(2,:));
h3 = plot(x, y1, 'linewidth', 3, 'Color', colors(1,:));
axis([x(1), x(end),-30, 0])
% axis([85, 95,-6, 0])
set(gca, 'fontsize', 20, 'linewidth', 2, 'xtick', 0:30:180, 'xticklabel', [-90:30:90])
% set(gca, 'fontsize', 20, 'linewidth', 2, 'xtick', 85:0.5:95)
xlabel('Angle (Elevation, \circ)', 'fontsize', 30)
ylabel('Normalized Power (dB)', 'fontsize', 30)
pbaspect([6,3,1])
legend([h3,h1,h2], {'AMS','16\times 1 Array',  '-3dB'}, 'location', 'northoutside', 'numcolumns', 3, 'box', 'off', 'fontsize', 30)
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/elevation_angle.eps')

bw1 = sum(y1 >= -3) / 10;
disp(sprintf("Elevation beamwidth (lens): %.1f degree", bw1))
% bw2 = sum(y2 >= -3);


figure(2)
clf
colors = colororder;
y1 = db(abs(azi_lens).^2) / 1.5;
y2 = db(abs(azi_array).^2) / 2;
h3 = plot(x, y2, 'linewidth', 3, 'Color', colors(3,:));
hold on
h2 = plot(xlim, [-3, -3], '--', 'linewidth', 3, 'Color', colors(2,:));
h1 = plot(x, y1, 'linewidth', 3, 'Color', colors(1,:));
axis([x(1), x(end),-30, 0])
% axis([85, 95,-6, 0])
set(gca, 'fontsize', 20, 'linewidth', 2, 'xtick', 0:30:180, 'xticklabel', [-90:30:90])
% set(gca, 'fontsize', 20, 'linewidth', 2, 'xtick', 85:95)
xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
ylabel('Normalized Power (dB)', 'fontsize', 30)
pbaspect([6,3,1])
legend([h1,h3,h2], {'AMS', '9\times 1 Array',  '-3dB'}, 'location', 'northoutside', 'numcolumns', 3, 'box', 'off', 'fontsize', 30)
% saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/azimuth_angle.eps')

bw1 = sum(y1 >= -3) / 10;
disp(sprintf("Azimuth beamwidth (lens): %.1f degree", bw1))

bw2 = sum(y2 >= -3) / 10;
disp(sprintf("Azimuth beamwidth (array): %.1f degree", bw2))