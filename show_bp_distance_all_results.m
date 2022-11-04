addpath('functions')
addpath('mat')
clear
for pic_ni = 0 : 10 : 8774
% load mat/optimal_200.mat
load(sprintf("gif/optimal_%i.mat", pic_ni))
load parameters.mat

theta = [0:1:180] / 180 * pi;

lens_num = 16;
c = 34300;
fc = 20e3;
lambda = c / fc;


lens_out = speaker_w * optG * diag(len_theta);


figure(1)
clf
subplot(231)
% plot((abs(bp(91,:)).^2), 'linewidth', 3)
imagesc(unwrap(angle(speaker_w), 2))
% imagesc(abs(speaker_w))
colorbar
pbaspect([1, 1, 1])
title("Speaker Phase")
ylabel("Scanning Angle")
xlabel("Speaker Index")

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

saveas(gcf, 'figs/matlab_test.png')

lens_angle = angle(len_theta);
lens_index = get_lens_index(len_theta);
save('optimal.mat', 'lens_angle', '-append')
save('optimal.mat', 'lens_index', '-append')

save_name = sprintf('gif/figs/results_%i.png', pic_ni);
saveas(gcf, save_name)
fprintf("process picture: %i\ni", pic_ni)

% w1 = speaker_w([31,111],:);

% a = sum(w1);
% lens_out = a * G * diag(len_theta);
% out = lens_out*steerVec;
% figure(4)
% clf
% plot(abs(out))
% saveas(gcf, 'figs/test.png')

end
