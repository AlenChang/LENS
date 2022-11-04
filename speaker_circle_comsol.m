
% clear

target_plane = 'yz2';
filename = strcat('comsol_data/old/6_speaker_beam_pattern.txt');

% data = read_comsol_data(filename);
data = readtable(filename);
data = table2array(data);
bp = data;
bpx = bp(:, 1:2);
bpx(:, 2) = bpx(:, 2) - min(bpx(:, 2));
bpangle = unwrap(acos(bpx(:, 1)./sqrt(bpx(:, 1).^2 + bpx(:,2).^2)));
a1 = linspace(pi, 0, 180);

% figure(2)
% clf
% plot(bpangle, 'linewidth')
% saveas(gcf, 'figs/matlab_test.png')

bp = bp(:, 3:end);
bpnew = zeros(180, 180);
for ti = 1 : 180
    bpnew(:, ti) = interp1(bpangle, bp(:, ti), a1, 'linear');
end
bp = bpnew;
bp = (abs(bp).^2.9)';

aa = bp / 100;
aa = aa / max(aa(:));
figure(1)
clf
imagesc(aa)
% hold on
% plot(1:181, 1:181, 'r*')
set(gca, 'xtick', 1:30:180, 'ytick', 1:30:180)
set(gca, 'fontsize', 20)
xlabel('Measured Angle', 'fontsize', 30)
ylabel('Target Angle', 'fontsize', 30)
colorbar('Ticks', 0:0.2:1)
pbaspect([1,1,1])
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/beam_pattern_comsol.eps')

gains = diag(bp);
gains = db(gains)/2.9;
gains = gains - max(gains);

amp2 = abs(diag(speaker_out)).^2 .* speaker_coeff;
amp2 = amp2 / max(amp2);
amp2 = db(amp2) - 9.5;

figure(2)
clf
h1 = plot(gains, 'linewidth', 3);
hold on
h2 = plot(xlim, [-3, -3], '--', 'linewidth', 3);
h3 = plot(amp2, 'linewidth', 3);
set(gca, 'fontsize', 20, 'linewidth', 2, 'xtick', 0:30:180, 'xticklabel', -90:30:90)
axis([0,180,-20,0])
xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
ylabel('Normalized Power (dB)', 'fontsize', 30)
pbaspect([6,3,1])
legend([h1,h3,h2], 'w/ lens', 'w/o lens', '-3dB', 'location', 'northoutside', 'numcolumns', 3, 'box', 'off', 'fontsize', 30)
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/diag_gain_change.eps')

figure(3)
clf
data1 = aa(91, :) / max(aa(91, :));
data2 = speaker_out(91, :) / max(speaker_out(91, :));
data3 = beams(91, :) / max(beams(91, :));
plot(db(data1), 'linewidth', 2)
hold on
plot(xlim, [-3, -3], '--r', 'linewidth', 2)
plot(db(data2.^2), 'linewidth', 2)
plot(db(data3.^2), 'linewidth', 2)
axis([70, 110, -10, 0])
% saveas(gcf, 'figs/matlab_test.png')

