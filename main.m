addpath('./functions/')
% clc
close all

%% define speakers
num_speakers = 9;
speaker_center = [0, 0];
speaker_spacing = 0.5;
speaker = build_speakers(num_speakers, speaker_center, speaker_spacing);

%% define lens

field_len2 = 30;
gridsize2 = speaker.lambda / 30;
field2 = build_sound_field(field_len2, gridsize2);

field_len = 20;
gridsize = speaker.lambda / 50;
field = build_sound_field(field_len, gridsize);

num_cells = 16;
lens_center = [-field_len2/2, 0];
focusing_point = [-field_len2/2+10, 4];
lens_spacing = 0.5;
lens = build_speakers(num_cells, lens_center, lens_spacing);
lens = backstepping(lens, focusing_point);
lens.weights = conj(lens.weights);

%% define targets
num_target =120;
target_coordinates = "square";
target = build_target_lens(field, target_coordinates, lens);

%% define desired sound pressure at samples
% target = desired_pressure_at_samples(target, speaker);

target = desired_pressure_at_samples_lens(target, lens);


%% solve SFR model
speaker = SRF_solution(target, speaker);

%% Compute sound field

field = compute_field_pressure(field, speaker);

tmp = lens;
tmp.locs(:, 1) = target.locs(round(end/2), 1);
lens.weights = getCoordinates(field, tmp);
field2 = compute_field_pressure(field2, lens);


%% Visulization
figure_init('GENERATED SOUND FIELD')
imagesc(field.x, field.y, real(field.sound_pressure))
hold on
plot(target.locs(:, 2), target.locs(:, 1), 'ro')
plot(speaker.locs(:, 2), speaker.locs(:, 1), 'r*')
set(gca, 'XTick', [], 'YTick', [])
xlabel('y')
ylabel('x')
pbaspect([1, 1, 1])



out = getCoordinates(field, target);

figure_init('COMPARE RESULTS AMPLITUDE')
plot(abs(target.sound_pressure), 'linewidth', 2)
hold on
plot(abs(out), 'linewidth', 2)

legend('Desired', 'Estimated', 'box', 'off', 'location', 'northeast', 'fontsize', 20)
pbaspect([1,1,1])

title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
set(gca, 'fontsize', 20)


figure_init('COMPARE RESULTS ANGLE')
plot(angle(target.sound_pressure), 'linewidth', 2)
hold on
plot(angle(out), 'linewidth', 2)
legend('Desired', 'Estimated', 'box', 'off', 'location', 'northeast', 'fontsize', 20)
pbaspect([1,1,1])

title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
set(gca, 'fontsize', 20)

figure_init('LENS SOUND FIELD')
imagesc(field2.x, field2.y, abs(field2.sound_pressure))
hold on
plot(lens.focusing_point(:, 2), lens.focusing_point(:, 1), 'ro')
plot(lens.locs(:, 2), lens.locs(:, 1), 'r*')
set(gca, 'XTick', [], 'YTick', [])
xlabel('y')
ylabel('x')
pbaspect([1, 1, 1])









