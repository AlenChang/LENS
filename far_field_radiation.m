clear
load optimal.mat
load parameters.mat

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

speaker_locs = zeros(6, 3);
speaker_locs(:,1) = (- (6 - 1):2:6 - 1) * lambda / 4;
speaker_weights = zeros(181, 6);
for ti = 1 : 181
    for mi = 1 : 6
        speaker_weights(ti, mi) = exp(-1j*2*pi*fc/c*(6-mi)*lambda/2*cos((ti-1)/180*pi));
    end
end

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

lens_out = speaker_w * G * diag(len_theta);
bp = zeros(length(distance), length(theta));

%% 2D results

% dxref = lens_locs2D(1, 1);
direction = 61;
[rx1, ele1] = helperPlotFarfield(lens_locs2D, lens_out, direction);

[rx2, ele2] = helperPlotFarfield(speaker_locs, speaker_weights, direction);

figure(5)
clf
subplot(221)
imagesc(ele1, ele1, abs(rx1))
xlabel("Elevation")
ylabel("Azimuth")
pbaspect([1,1,1])

subplot(222)
imagesc(ele1, ele1, abs(rx2))
xlabel("Elevation")
ylabel("Azimuth")
pbaspect([1,1,1])

subplot(223)
bp = reshape(lens_out(direction,:), 16, 16);
imagesc(unwrap(angle(bp)))
pbaspect([1,1,1])

subplot(224)
% plot(lens_locs2D(:,1), lens_locs2D(:,2), '*')
plot(abs(rx1(round(end/2), :)).^2, 'linewidth', 2)
pbaspect([1,1,1])
saveas(gcf, 'figs/test.png')

figure(2)
clf
a1 = rx1(round(end/2), :);
amax = max(abs(a1));
a1 = a1 / amax;
a2 = rx2(round(end/2), :);
a2 = a2 / amax;
hold on
plot(db(abs(a1).^2), 'linewidth', 3)
plot(db(abs(a2).^2), 'linewidth', 3)
axis([1, length(ele1), -80, 0])
xlabel("Direction")
ylabel("Gain (dB)")
set(gca, 'fontsize', 20, 'box', 'on', 'linewidth', 2)
legend('w/ LENS', 'w/o LENS', 'box', 'off', 'fontsize', 30)
pbaspect([6,4,1])
saveas(gcf, 'figs/power.eps')



function [rx1, elevation_theta] = helperPlotFarfield(lens_locs2D, lens_out, direction)
    r = 1e6;
    c = 34300;
    fc = 20e3;
    
    lens_num = size(lens_locs2D, 1);

    % dmat = zeros(181, 181);
    elevation_theta = 30:150;
    sample_points = length(elevation_theta);
    rx1 = zeros(sample_points^2, 1);
    observe_plane = zeros(sample_points^2, 3);
    square_len = r * cos(elevation_theta(1)/180*pi);

    sample_x = linspace(-square_len, square_len, sample_points);
    for ti = 1 : sample_points
        for mi = 1 : sample_points
            observe_plane((ti-1)*sample_points+mi, 1) = sample_x(ti);
            observe_plane((ti-1)*sample_points+mi, 2) = sample_x(mi);
            % observe_plane((ti-1)*sample_points+mi,3) = sqrt(r^2 - sample_x(ti)^2 - sample_x(mi)^2);
        end
    end
    observe_plane(:,3) = r * sin(elevation_theta(1)/180*pi);

    for mi = 1 : size(observe_plane, 1)
        target = observe_plane(mi, :);
        for ti = 1:lens_num
            % numel(lens_locs2D(ti, :))
            % numel(target)
            d = compute_distance(target, lens_locs2D(ti, :));
            
            % dmat(idx, idy) = d;
            rx1(mi) = rx1(mi) + lens_out(direction, ti) * exp(-1j * 2 * pi * fc / c * d) / d;
        end

    end

    rx1 = reshape(rx1, sample_points, sample_points);
end