clear
load optimal.mat
load parameters.mat

distance = 10:1:200;
theta = [0:1:180] / 180 * pi;

c = 34300;
fc = 20e3;
lambda = c / fc;
lens_locs = (-15:2:15) * lambda / 4;
lens_locs = [lens_locs' zeros(16, 1)];

if (length(len_theta) == 8)
    len_theta = [len_theta; flipud(len_theta)];
end

lens_out = speaker_w * G * diag(len_theta);

bp = zeros(length(distance), length(theta));

for ti = 1:length(distance)

    for zi = 1:length(theta)
        out = 0;
        target_locs = [distance(ti) * cos(theta(zi)), distance(ti) * sin(theta(zi))];

        for mi = 1:16
            d = compute_distance(target_locs, lens_locs(mi, :));
            out = out + lens_out(91, mi) * exp(-1j * 2 * pi * fc / c * d);
        end

        bp(ti, zi) = out;
    end

    % disp(sprintf('%i out of %i\n', ti, length(distance)))
end

figure(1)
clf
subplot(231)
imagesc(theta / pi * 180, distance / 100, abs(bp).^2)
set(gca, 'fontsize', 10, 'YDir', 'reverse')
xlabel('Angle')
ylabel('Distance')
title("Energy VS Distance (Power)")
pbaspect([1, 1, 1])
% colorbar

subplot(232)
% plot((abs(bp(91,:)).^2), 'linewidth', 3)
imagesc(unwrap(angle(speaker_w), 2))
pbaspect([1, 1, 1])
title("Speaker Phase")
ylabel("Scanning Angle")
xlabel("Speaker Index")
% xlable("Angle")
% ylable("Power")

subplot(234)
beams = lens_out * steerVec;
imagesc(abs(beams).^2)
xlabel("Angle")
ylabel("Scanning Angle")
title("Beam Patterns (Power)")
pbaspect([1, 1, 1])
% colorbar

subplot(235)
beams_amp = abs(beams(31:10:151, :))';
plot(beams_amp.^2, 'linewidth', 2)
pbaspect([1, 1, 1])

subplot(233)
imagesc(abs(speaker_w))
title("Speaker Amplitude")
ylabel("Scanning Angle")
xlabel("Speaker Index")
pbaspect([1, 1, 1])

subplot(236)
plot(unwrap(angle(len_theta)), 'linewidth', 3)
xlabel("LENS index")
ylabel("Phase delay")
title("LENS design")
pbaspect([1, 1, 1])

saveas(gcf, 'figs/matlab_test.png')

lens_angle = angle(len_theta);
lens_index = get_lens_index(len_theta);
save('optimal.mat', 'lens_angle', '-append')
save('optimal.mat', 'lens_index', '-append')
