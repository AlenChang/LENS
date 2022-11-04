addpath('functions')
addpath('mat')

clear

% load mat/optimal_200.mat
load optimal_non_uniform_2.mat
load parameters_non_uniform_2.mat

distance = 10:1:200;
theta = [0:1:180] / 180 * pi;

lens_num = 16;
c = 34300;
fc = 20e3;
lambda = c / fc;

lens_locs = (- (lens_num - 1):2:lens_num - 1) * lambda / 4;
lens_locs = [lens_locs' zeros(lens_num, 1)];

lens_locs_x = (- (lens_num - 1):2:lens_num - 1) * lambda / 4;
lens_locs_y = lens_locs_x;

len_1D = length(lens_locs_x);
lens_locs2D = zeros(len_1D^2, 3);

for ti = 1:len_1D

    for mi = 1:len_1D
        lens_locs2D((ti - 1) * len_1D + mi, 2) = lens_locs_x(ti);
        lens_locs2D((ti - 1) * len_1D + mi, 1) = lens_locs_x(mi);
        lens_locs2D((ti - 1) * len_1D + mi, 3) = 0;
    end

end

% if (strcmp(lens_dimension, "1D"))
%     len_theta = [len_theta; flipud(len_theta)];
% end

% win = kaiser(6, 1);
% speaker_w = win' .* speaker_w;

lens_out = speaker_w * optG * diag(len_theta);
bp = zeros(length(distance), length(theta));

%% 2D results

% % dxref = lens_locs2D(1, 1);
% if (strcmp(lens_dimension, '2D'))
%     r = 1e3;
%     rx = zeros(181, 181);

%     for thetax = linspace(0, pi, 181)

%         for thetay = linspace(0, pi, 181)
%             target = zeros(1, 3);
%             target(1, 1) = r * cos(thetax) * cos(thetay);
%             target(:, 2) = r * cos(thetax) * sin(thetay);
%             target(:, 3) = r * sin(thetax);

%             for ti = 1:256
%                 d = compute_distance(target, lens_locs2D(ti, :));
%                 rx(round(thetax / pi * 180 + 1), round(thetay / pi * 180 + 1)) = lens_out(91, ti) * exp(-1j * 2 * pi * fc / c * d);
%             end

%         end

%     end

% end

% figure(5)
% clf
% imagesc(abs(rx))
% saveas(gcf, 'figs/test.png')

%% 1D results
for ti = 1:length(distance)

    for zi = 1:length(theta)
        out = 0;
        target_locs = [distance(ti) * cos(theta(zi)), distance(ti) * sin(theta(zi))];

        for mi = 1:lens_num
            d = compute_distance(target_locs, lens_locs(mi, :));
            out = out + lens_out((end+1)/2, mi) * exp(-1j * 2 * pi * fc / c * d) / (2 * pi * d);
            % out = out + lens_out((end+1)/2, mi) * exp(-1j * 2 * pi * fc / c * d);
        end

        bp(ti, zi) = out;
    end

    % disp(sprintf('%i out of %i\n', ti, length(distance)))
end

figure(11)
clf
data11 = db(abs(bp));
data11 = flipud(data11);
data11 = data11 - max(data11(:));
thre = -50;
data11(data11 < thre) = thre;
imagesc(theta / pi * 180, distance / 100, data11)
hold on
plot(xlim,1.8*[1, 1], '--r', 'linewidth', 3)
set(gca, 'fontsize', 10)
set(gca, 'fontsize', 20, 'xtick', [0:30:180], 'xticklabel', [-90:30:90])
set(gca, 'ytick', linspace(distance(1)/100, distance(end)/100, 3), 'yticklabel', [2:-1:0])
xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
ylabel('Distance (m)', 'fontsize', 30)
pbaspect([6, 3, 1])
colorbar
% title("Energy VS Distance (Power)")
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/convergence.eps')

   


figure(1)
clf
subplot(231)
imagesc(theta / pi * 180, distance / 100, db(abs(bp)))
set(gca, 'fontsize', 10, 'YDir', 'reverse')
xlabel('Angle')
ylabel('Distance')
title("Energy VS Distance (Power)")
pbaspect([1, 1, 1])
colorbar
% subplot(231)
% % plot((abs(bp(91,:)).^2), 'linewidth', 3)
% imagesc(unwrap(angle(speaker_w), 2))
% % imagesc(abs(speaker_w))
% colorbar
% pbaspect([1, 1, 1])
% title("Speaker Phase")
% ylabel("Scanning Angle")
% xlabel("Speaker Index")

subplot(232)
% plot((abs(bp(91,:)).^2), 'linewidth', 3)
% imagesc(unwrap(angle(speaker_w), 2))
imagesc(abs(speaker_w))
colorbar
pbaspect([1, 1, 1])
title("Speaker Magnitude")
ylabel("Scanning Angle")
xlabel("Speaker Index")
% xlable("Angle")
% ylable("Power")

subplot(234)
beams = lens_out * steerVec;
x = 30:150;
imagesc(x, x, abs(beams).^2)
set(gca, 'xtick', 30:30:150, 'ytick', 30:30:150, 'xticklabel', -60:30:60, 'yticklabel', -60:30:60)
xlabel("Angle")
ylabel("Scanning Angle")
title("Beam Patterns (Power)")
pbaspect([1, 1, 1])
% colorbar

subplot(235)
beams_amp = abs(beams(1:10:121, :))';
plot(beams_amp.^2, 'linewidth', 2)
% axis([0, 181, 0, 40])
pbaspect([1, 1, 1])

% subplot(233)
% imagesc(abs(speaker_w))
% title("Speaker Amplitude")
% ylabel("Scanning Angle")
% xlabel("Speaker Index")
% pbaspect([1, 1, 1])

subplot(233)
aa = abs(beams((end+1)/2, :));
aa = aa / max(aa);
aa = db(aa);
index = aa >= -3;
beam_width = sum(index);

a2 = abs(beams(11, :));
a2 = a2 / max(a2);
a2 = db(a2);

plot(aa, 'linewidth', 2)
hold on
plot(xlim, [-3, -3], 'r--', 'linewidth', 2)
plot(a2, 'linewidth', 2)
axis([0, 121, -20, 0])
title(sprintf('Beamwidth = %i', beam_width))
pbaspect([1, 1, 1])


subplot(236)
lens_design = unwrap(angle(len_theta));

if (strcmp(lens_dimension, "2D"))
    imagesc(reshape(lens_design, 16, 16))
else
    plot(lens_design, 'linewidth', 3)
    xlabel("LENS index")
    ylabel("Phase delay")
end

title("LENS design")
pbaspect([1, 1, 1])

% saveas(gcf, 'figs/matlab_test.png')

lens_angle = angle(len_theta);
lens_index = get_lens_index(len_theta);
save('optimal.mat', 'lens_angle', '-append')
save('optimal.mat', 'lens_index', '-append')

beams_amp = abs(beams)';
figure(3)
clf
plot(beams_amp(:, 7).^2, 'linewidth', 4)
saveas(gcf, 'figs/matlab_power.png')

figure(4)
clf
subplot(331)
plot(beams_amp(:, 31).^2, 'linewidth', 3)
title("30")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

subplot(332)
plot(beams_amp(:, 41).^2, 'linewidth', 3)
title("40")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

subplot(333)
plot(beams_amp(:, 51).^2, 'linewidth', 3)
title("50")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

subplot(334)
plot(beams_amp(:, 61).^2, 'linewidth', 3)
title("60")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

subplot(335)
plot(beams_amp(:, 71).^2, 'linewidth', 3)
title("70")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

subplot(336)
plot(beams_amp(:, 81).^2, 'linewidth', 3)
title("80")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

subplot(337)
plot(beams_amp(:, 91).^2, 'linewidth', 3)
title("90")
set(gca, 'fontsize', 20)
% pbaspect([1,1,1])

saveas(gcf, 'figs/matlab_beams.png')

% w1 = speaker_w([31,111],:);

% a = sum(w1);
% lens_out = a * G * diag(len_theta);
% out = lens_out*steerVec;
% figure(4)
% clf
% plot(abs(out))
% saveas(gcf, 'figs/test.png')
