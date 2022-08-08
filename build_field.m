function [field_speaker, field_lens, target, lens, speaker] = build_field(num_speakers, sweep_angle, plot_code, parameter_source)
%% define speakers
% num_speakers = 9;
speaker_center = [0, 0];
speaker_spacing = 0.5;
fc = 20e3;
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
steering_angle = sweep_angle; % (-90, 90)
focusing_point = [-field_len_x / 2 + 10, 10 * tan(steering_angle / 180 * pi)];
lens_spacing = 0.5;
lens = build_speakers(num_cells, lens_center, lens_spacing, fc);

lens = get_lens_delay(lens, speaker, field_speaker);

if (strcmp(parameter_source, "random"))
    lens.delay = lens_delay;
    lens.back_step = lens.delay;
elseif (strcmp(parameter_source, "optimal"))
    load optimal.mat
    lens.delay = len_theta;
    lens.back_step = lens.delay;
elseif(strcmp(parameter_source, "raw"))

else
    error("Wrong parameter source type!")
end

% lens = compensate_phase_shift_frequency(lens);

focusing_type = 'direction'; % {'direction', 'point'}
lens = backstepping(lens, focusing_point, focusing_type);
% lens.weights = conj(lens.weights);

%% define targets
target_coordinates = "square";
target = build_target_lens(field_speaker, target_coordinates, lens);

%% define desired sound pressure at samples
% target = desired_pressure_at_samples(target, speaker);

target = desired_pressure_at_samples_lens(target, lens);

%% solve SFR model
speaker = SRF_solution(target, speaker);

if (strcmp(parameter_source, "optimal"))
    speaker.weights_out = speaker_w(plot_code, :) ./ sum(abs(speaker_w(plot_code, :)));
end

%% Compute sound field

field_speaker = compute_field_pressure(field_speaker, speaker);

% keyboard
tmp = lens;
tmp.locs(:, 1) = target.locs(round(end / 2), 1);
lens.weights_in = getCoordinates(field_speaker, tmp);
% lens.weights_in = lens.weights_in / sum(abs(lens.weights_in));
lens.weights_out = lens.weights_in .* lens.delay;
field_lens = compute_field_pressure(field_lens, lens);


end