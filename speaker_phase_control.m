addpath('./functions/')
% clc
% close all

%% STEP1 define speakers
num_speakers = 9;
speaker_center = [0, 0];
speaker_spacing = 0.5; % 0.5 *lambda
fc = 20e3;
speaker = build_speakers(num_speakers, speaker_center, speaker_spacing, fc);

%% STEP2 define targets
num_target =120;
target_coordinates = "circle";
target = build_target(num_target, target_coordinates);

%% STEP3 define desired sound pressure at samples
target = desired_pressure_at_samples(target, speaker);

%% STEP4 solve SFR model
speaker = SRF_solution(target, speaker);

%% STEP5 Compute sound field
field_len = 20;
gridsize = speaker.lambda / 50;
field = build_sound_field(field_len, field_len, gridsize);
field = compute_field_pressure(field, speaker);


%% STEP6 Visulization
figure_init('GENERATED SOUND FIELD')
imagesc(field.x, field.y, db(abs(field.sound_pressure)))
hold on
plot(target.locs(:, 2), target.locs(:, 1), 'ro')
plot(speaker.locs(:, 2), speaker.locs(:, 1), 'r*')
set(gca, 'XTick', [], 'YTick', [])
xlabel('y')
ylabel('x')
pbaspect([1, 1, 1])
saveas(gcf, 'figs/speaker_generated_sound_field.png')



out = getCoordinates(field, target);
x = target.phi / pi * 180;

figure_init('COMPARE RESULTS AMPLITUDE')
plot(x, abs(target.sound_pressure), 'linewidth', 2)
hold on
plot(x, abs(out), 'linewidth', 2)
target_angle = plot_max_value(x, target.sound_pressure);
real_angle = plot_max_value(x, out);
legend('Desired', 'Estimated', 'box', 'off', 'location', 'northeast', 'fontsize', 20)
pbaspect([1,1,1])

title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
set(gca, 'XTick', [0, 30, 60, 90, 120, 150, 180], 'fontsize', 20)

figure_init('COMPARE RESULTS ANGLE')
plot(x, angle(target.sound_pressure), 'linewidth', 2)
hold on
plot(x, angle(out), 'linewidth', 2)










