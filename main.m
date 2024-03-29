addpath('./functions/')
% clc
close all

%% define speakers
num_speakers = 4;
speaker_center = [0, 0];
speaker_spacing = 0.5;
fc = 19e3;
speaker = build_speakers(num_speakers, speaker_center, speaker_spacing, fc);

%% define lens

field_len2 = 50;
gridsize2 = speaker.lambda / 20;
field_lens = build_sound_field(field_len2, field_len2, gridsize2);

field_len_x = 20;
field_len_y = 20;
gridsize = speaker.lambda / 50;
field_speaker = build_sound_field(field_len_x, field_len_y, gridsize);

num_cells = 16;
lens_center = [-field_len2/2, 0];
steering_angle = 0; % (-90, 90)
focusing_point = [-field_len2/2+10, 10 * tan(steering_angle / 180 * pi)];
lens_spacing = 0.5;
lens = build_speakers(num_cells, lens_center, lens_spacing, 20e3);

lens = get_lens_delay(lens, speaker, field_speaker);
focusing_type = 'direction'; % {'direction', 'point'}
lens = backstepping(lens, focusing_point, focusing_type);
% lens.weights = conj(lens.weights);

%% define targets
num_target =120;
target_coordinates = "square";
target = build_target_lens(field_speaker, target_coordinates, lens);

%% define desired sound pressure at samples
% target = desired_pressure_at_samples(target, speaker);

target = desired_pressure_at_samples_lens(target, lens);


%% solve SFR model
speaker = SRF_solution(target, speaker);

%% Compute sound field

field_speaker = compute_field_pressure(field_speaker, speaker);

tmp = lens;
tmp.locs(:, 1) = target.locs(round(end/2), 1);
lens.weights = getCoordinates(field_speaker, tmp);
lens.weights = lens.weights .* lens.delay;
field_lens = compute_field_pressure(field_lens, lens);


%% Visulization
% figure_init('GENERATED SOUND FIELD')
figure(1)
set(gcf, 'Position', [210.6000 69.8000 1.0934e+03 980.2000])
clf
subplot(221)
imagesc(field_speaker.y, field_speaker.x, real(field_speaker.sound_pressure))
hold on
plot(target.locs(:, 2), target.locs(:, 1), 'ro')
plot(speaker.locs(:, 2), speaker.locs(:, 1), 'r*')
plot(lens.locs(:, 2), field_speaker.x(end), 'gs', 'MarkerFaceColor','g')
set(gca, 'XTick', [], 'YTick', [])
xlabel('y')
ylabel('x')
pbaspect([field_speaker.len_y, field_speaker.len_x, 1])



out = getCoordinates(field_speaker, target);


% figure_init('COMPARE RESULTS AMPLITUDE')
subplot(222)
plot(abs(target.sound_pressure), 'linewidth', 2)
hold on
plot(abs(out), 'linewidth', 2)

legend('Desired', 'Estimated', 'box', 'off', 'location', 'northeast', 'fontsize', 20)
pbaspect([1,1,1])

% title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
set(gca, 'fontsize', 20)


% figure_init('COMPARE RESULTS ANGLE')
subplot(224)
plot(angle(target.sound_pressure), 'linewidth', 2)
hold on
plot(angle(out), 'linewidth', 2)
legend('Desired', 'Estimated', 'box', 'off', 'location', 'northeast', 'fontsize', 20)
pbaspect([1,1,1])

% title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
set(gca, 'fontsize', 20)

% figure_init('LENS SOUND FIELD')
subplot(223)
imagesc(field_lens.x, field_lens.y, abs(field_lens.sound_pressure))
hold on
plot(lens.focusing_point(:, 2), lens.focusing_point(:, 1), 'ro')
plot(lens.locs(:, 2), lens.locs(:, 1), 'gs', 'MarkerFaceColor','g')
set(gca, 'XTick', [], 'YTick', [])
title(sprintf('Steering Angle: %.2f', steering_angle), 'FontSize', 20)
xlabel('y')
ylabel('x')
pbaspect([1, 1, 1])









