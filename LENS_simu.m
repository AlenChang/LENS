%% STEP1 Define field
field_len = 40;
gridsize = 0.1;
field2 = build_sound_field(field_len, field_len, gridsize);

%% STEP1 Define LENS locs
clc
num_lens = 16;
lens_center = [-field_len/2, 0];
% focusing_point = [-field_len/2+10, 4];
steering_angle = 10; % (-90, 90)
focusing_point = [-field_len/2+10, 10 * tan(steering_angle / 180 * pi)];
lens_spacing = 0.5;
fc = 20e3;
lens = build_speakers(num_lens, lens_center, lens_spacing, fc);
focusing_type = 'direction'; % {'direction', 'point'}
% lens.delay = ones(lens.num, 1);
lens = get_lens_delay(lens, speaker, field_speaker);
lens = backstepping(lens, focusing_point, focusing_type);
% lens.weights = conj(lens.weights);

%% STEP2 Compute sound field

field2 = compute_field_pressure(field2, lens);


%% STEP3 Visulization
figure_init('LENS SOUND FIELD')
imagesc(field2.x, field2.y, abs(field2.sound_pressure))
hold on
plot(lens.focusing_point(:, 2), lens.focusing_point(:, 1), 'ro')
plot(lens.locs(:, 2), lens.locs(:, 1), 'r*')
set(gca, 'XTick', [], 'YTick', [])
xlabel('y')
ylabel('x')
pbaspect([1, 1, 1])

figure_init('LENS PHASE')
plot(unwrap(angle(lens.weights)), 'LineWidth', 2)
xlabel('lens index')
ylabel('Phase (rad)')
set(gca, 'FontSize', 20)