addpath('./functions/')
% clc
close all

% freq = 10e3:500:30e3;
freq = 20e3;
sweep_angle = [-40:5:40];
amps = zeros(size(freq));
cmap = jet;
index = round(linspace(1, size(cmap, 1), length(sweep_angle)));
cmap = cmap(index, :);

rng(1)
num_test = 1;
amps_total = zeros(num_test, 1);
A = zeros(length(sweep_angle), 16);

lens_delay = exp(1j * rand(16, 1) * 2 * pi);
close all

amps_record = zeros(length(sweep_angle), 1);

for zi = 1:1
    %% define speakers
    num_speakers = 9;
    speaker_center = [0, 0];
    speaker_spacing = 0.5;
    fc = freq;
    speaker = build_speakers(num_speakers, speaker_center, speaker_spacing, fc);

    %% define lens

    field_len_x = 20;
    field_len_y = 20;
    gridsize2 = speaker.lambda / 20;
    field_lens = build_sound_field(field_len_x, field_len_y, gridsize2);

    field_speaker_x = 10;
    field_speaker_y = 20;
    gridsize = speaker.lambda / 20;
    field_speaker = build_sound_field(field_speaker_x, field_speaker_y, gridsize);

    num_cells = 16;
    lens_center = [-field_len_x / 2, 0];
    steering_angle = 0; % (-90, 90)
    focusing_point = [-field_len_x / 2 + 10, 10 * tan(steering_angle / 180 * pi)];
    lens_spacing = 0.5;
    lens = build_speakers(num_cells, lens_center, lens_spacing, fc);
    lens = get_lens_delay(lens, speaker, field_speaker);

    focusing_type = 'direction'; % {'direction', 'point'}
    lens = backstepping(lens, focusing_point, focusing_type);
    % lens.weights = conj(lens.weights);

    %% define targets
    target_coordinates = "raw";
    target = build_target_lens(field_speaker, target_coordinates, lens);
    target.locs = lens.locs;
    target.locs(:, 1) = field_speaker.x(end);
    target.sound_pressure = lens.weights_out;
    target.num = 16;

    %% define desired sound pressure at samples
    % target = desired_pressure_at_samples(target, speaker);

    % target = desired_pressure_at_samples_lens(target, lens);

    %% solve SFR model
    [speaker, G] = SRF_solution(target, speaker);
    G = transpose(G);
    % speaker.weights = exp(1j*pi/2);
    amps(zi) = abs(sum(lens.weights_out));
    [steerVec, theta] = getSteeringMatrix(lens);
    for ni = 1 : length(sweep_angle)
        target_angle = sweep_angle(ni);
        A(ni, :) = steerVec(:, target_angle+91);
    end

end

save parameters.mat A G

% locs = zeros(length(field_lens.x), 2);
% locs(:, 1) = field_lens.x;
% [x, y] = getCoordinatesCore(field_lens, locs);
% aa = field_lens.sound_pressure(x, y);
% figure(3)
% clf
% plot(abs(aa))
