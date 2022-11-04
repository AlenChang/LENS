addpath('./functions/')
clear
% load optimal.mat
% load parameters.mat
load optimal_non_uniform_2.mat
load parameters_non_uniform_2.mat

load steerVec.mat

% steerVec = steerVec / 1e6;

params.c = 34300;
params.fc = 20e3;
params.lambda = params.c / params.fc;

num_lens = size(optG, 2);
lens_locs = generate2Dlocs(params, num_lens);
lens_out = speaker_w * optG * diag(len_theta);
beams = lens_out * steerVec;

beams_all = lens_out * steerVecAll;

num_speaker = size(optG, 1);
speaker_locs = generate1Dlocs(params, num_speaker);
speaker_weights = getPhasedArraySteeringWeights(params, num_speaker);
speaker_weights = speaker_weights(31:151, :);

% speaker_weights = speaker_weights(31:121, :);

speaker_out = speaker_weights * steerVec(1:num_speaker,:);

speaker_out_all = speaker_weights * steerVecAll(1:num_speaker,:);

% speaker_out = speaker
speaker_power = sum(abs(speaker_out).^2, 2);
speaker_coeff = speaker_power((end+1) / 2) ./ speaker_power;

beams_power = sum(abs(beams).^2, 2);
beams_coeff = beams_power((end+1) / 2) ./ beams_power;
% compare_phase_array_lens
lens_power = sum(abs(lens_out).^2, 2);

figure(1)
clf
amp1 = abs(diag(beams)).^2 .* beams_coeff ./ lens_power;
amp2 = abs(diag(speaker_out)).^2 .* speaker_coeff;
% amp2 = amp2 - sqrt(max(abs(amp2)));
amp_max = max(abs(amp1));
amp1 = amp1 / amp_max;
amp2 = amp2 / amp_max;
x = 20:90;
plot(x, db(amp1(x+1)), 'linewidth', 3)
hold on
plot(x, db(amp2(x+1))-3, 'linewidth', 3)
set(gca, 'fontsize', 20)
axis([x(1), x(end), -20, 0])
legend('w/ lens', 'w/o lens', 'box', 'off', 'location', 'southeast', 'fontsize', 30)
xlabel('Angles', 'fontsize', 30)
ylabel('Normalized Power (dB)', 'fontsize', 30)
pbaspect([6,4,1])
set(gca, 'linewidth', 2)
saveas(gcf, 'figs/lens_gain.png')
print('-depsc', 'fig_results/lens_gain.eps')

% keyboard
figure(4)
clf
beams_power = abs(beams).^2;
beams_power = beams_power / max(max(beams_power));
imagesc(beams_power)
colorbar('Ticks', 0:0.2:1)
set(gca, 'fontsize', 20)
pbaspect([1, 1, 1])
xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
ylabel('Steering Angle (\circ)', 'fontsize', 30)
set(gca, 'xtick', 1:30:121, 'xticklabel', -60:30:60, ...
    'ytick', 1:30:121, 'yticklabel', -60:30:60)
saveas(gcf, 'figs/beam_pattern_results.png')
print('-depsc', 'fig_results/beam_pattern_results.eps')

figure(4)
clf
beams_power = abs(speaker_out).^2 .* speaker_coeff;
beams_power = beams_power / max(max(beams_power));
imagesc(beams_power)
colorbar
set(gca, 'fontsize', 20)
pbaspect([1, 1, 1])
xlabel("Measured Angle", 'fontsize', 30)
ylabel("Target Angle", 'fontsize', 30)
set(gca, 'xtick', 0:30:180, ...
    'ytick', 0:30:180, 'XTickLabelRotation', 25)
saveas(gcf, 'figs/speaker_pattern_results.png')
print('-depsc', 'fig_results/speaker_pattern_results.eps')

% keyboard

lens_phase = reshape(unwrap(angle(len_theta)), sqrt(num_lens), []);
% lens_phase = unwrap(angle(lens_phase));

aa_max = max(lens_phase(:));
figure(5)
clf
index_label = 0:15;
imagesc(index_label, index_label, lens_phase' - aa_max)
colorbar;
set(gca, 'fontsize', 20)
pbaspect([1, 1, 1])
xlabel("LENS index (X-axis)", 'fontsize', 30)
ylabel("LENS index (Y-axis)", 'fontsize', 30)
set(gca, 'xtick', index_label(1:2:end), 'ytick', index_label(1:2:end), 'XTickLabelRotation', 25, 'YTickLabelRotation', 25)
% set(gca, 'xticklabel', 1:16, 'ytick', 1:16)
saveas(gcf, 'figs/lens_phase.png')
print('-depsc', 'fig_results/lens_phase.eps')

lens_index = get_lens_index(len_theta);
lens_design = reshape(lens_index, sqrt(num_lens), []);
figure(6)
clf
index_label = 0:15;
C = hot(16);
colormap(flipud(C))
lens_design = mod(lens_design + 6, 16);
imagesc(index_label, index_label, lens_design')
colorbar('Ticks', 0:5:15);

set(gca, 'fontsize', 20)
pbaspect([1, 1, 1])
xlabel("LENS index (X-axis)", 'fontsize', 30)
ylabel("LENS index (Y-axis)", 'fontsize', 30)
set(gca, 'xtick', index_label(1:2:end), 'ytick', index_label(1:2:end), 'XTickLabelRotation', 25, 'YTickLabelRotation', 25)
% set(gca, 'xticklabel', 1:16, 'ytick', 1:16)
saveas(gcf, 'figs/lens_design.png')
print('-depsc', 'fig_results/lens_design.eps')

figure(6)
colormap(parula)
imagesc(abs(speaker_w))
colorbar
set(gca, 'fontsize', 20, 'ytick', 1:60:181, 'yticklabel', -60:60:60, 'xtick', 1:num_speaker)
xlabel("Speaker Index", 'fontsize', 30)
ylabel("Target Angles", 'fontsize', 30)
pbaspect([1, 1, 1])

saveas(gcf, 'figs/codebook_amp.png')
print('-depsc', 'fig_results/codebook_amp.eps')

figure(7)
colormap(parula)
imagesc(angle(speaker_w .* conj(speaker_w(:, 1))))
colorbar
set(gca, 'fontsize', 20, 'ytick', 1:60:181, 'yticklabel', -60:60:60, 'xtick', 1:num_speaker)
xlabel("Speaker Index", 'fontsize', 30)
ylabel("Target Angles", 'fontsize', 30)
pbaspect([1, 1, 1])

saveas(gcf, 'figs/codebook_phase.png')
print('-depsc', 'fig_results/codebook_phase.eps')

%% 2D results

direction = [31, 61, 91]-30;
elevation_theta = [30:150];

for target_angle = direction
    rx1 = helperPlotFarfield(params, lens_locs, lens_out(target_angle, :), elevation_theta);


    rx2 = helperPlotFarfield(params, speaker_locs, speaker_weights(target_angle, :), elevation_theta);

    figure(5)
    clf
    imagesc(abs(rx1).^2)
    xlabel("Azimuth")
    ylabel("Elevation")
    set(gca, 'fontsize', 30, 'xticklabel', [], 'yticklabel', [])
    pbaspect([1, 1, 1])
    saveas(gcf, sprintf('figs/2D_compare_lens_%i.png', target_angle-1+30))
    print('-depsc', sprintf('fig_results/2D_compare_lens_%i.eps', target_angle-1+30))

    rx2 = rx2 ./ max(abs(rx2(:)));
    figure(5)
    clf
    imagesc(abs(rx2).^2)
    xlabel("Azimuth")
    ylabel("Elevation")
    set(gca, 'fontsize', 30, 'xticklabel', [], 'yticklabel', [])
    pbaspect([1, 1, 1])
    % colorbar
    saveas(gcf, sprintf('figs/2D_compare_array_%i.png', target_angle-1+30))
    print('-depsc', sprintf('fig_results/2D_compare_array_%i.eps', target_angle-1+30))
end

% subplot(223)
% bp = reshape(lens_out(direction, :), 16, 16);
% imagesc(unwrap(angle(bp)))
% pbaspect([1, 1, 1])

% subplot(224)
% % plot(lens_locs(:,1), lens_locs(:,2), '*')
% plot(abs(rx1(round(end / 2), :)).^2, 'linewidth', 2)
% pbaspect([1, 1, 1])
% saveas(gcf, 'figs/test.png')

figure(2)
clf
a1 = rx1(round(end / 2), :);
amax = max(abs(a1));
a1 = a1 / amax;
a2 = rx2(round(end / 2), :);
a2 = a2 / amax;
hold on
plot(db(abs(a1).^2), 'linewidth', 3)
plot(db(abs(a2).^2), 'linewidth', 3)
axis([1, length(elevation_theta), -80, 0])
xlabel("Direction")
ylabel("Gain (dB)")
set(gca, 'fontsize', 20, 'box', 'on', 'linewidth', 2)
legend('w/ LENS', 'w/o LENS', 'box', 'off', 'fontsize', 30)
pbaspect([6, 4, 1])
saveas(gcf, 'figs/power.eps')


% show_non_diagonal_peaks

% visualize_sound_pressure


% comapre_elevation_beamwidth

