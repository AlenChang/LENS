addpath('./functions/')
% clc
close all

% freq = 10e3:500:30e3;
freq = 20e3;
sweep_angle = [-60:5:60];
amps = zeros(size(freq));
cmap = jet;
index = round(linspace(1, size(cmap, 1), length(sweep_angle)));
cmap = cmap(index, :);

parameter_source = "optimal";
% parameter_source = "raw";

rng(1)
num_test = 1;
amps_total = zeros(num_test, 1);
for mi = 1:num_test

    lens_delay = exp(1j * rand(16, 1) * 2 * pi);
    % close all

    amps_record = zeros(length(sweep_angle), 1);
    for zi = 1:length(sweep_angle)
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
        steering_angle = sweep_angle(zi); % (-90, 90)
        focusing_point = [-field_len_x / 2 + 10, 10 * tan(steering_angle / 180 * pi)];
        lens_spacing = 0.5;
        lens = build_speakers(num_cells, lens_center, lens_spacing, fc);

        lens = get_lens_delay(lens, speaker, field_speaker);
        if(strcmp(parameter_source, "random"))
            lens.delay = lens_delay;
            lens.back_step = lens.delay;
        elseif(strcmp(parameter_source, "optimal"))
            load optimal.mat len_theta
            lens.delay = len_theta;
            lens.back_step = lens.delay;
        end

        % lens = compensate_phase_shift_frequency(lens);

        focusing_type = 'direction'; % {'direction', 'point'}
        lens = backstepping(lens, focusing_point, focusing_type);
        % lens.weights = conj(lens.weights);

        %% define targets
        num_target = 120;
        target_coordinates = "square";
        target = build_target_lens(field_speaker, target_coordinates, lens);

        %% define desired sound pressure at samples
        % target = desired_pressure_at_samples(target, speaker);

        target = desired_pressure_at_samples_lens(target, lens);

        %% solve SFR model
        speaker = SRF_solution(target, speaker);
        if(strcmp(parameter_source, "optimal"))
            load optimal.mat speaker_w
            speaker.weights_out = speaker_w(zi, :);
        end


        % speaker.weights = exp(1j*pi/2);

        %% Compute sound field

        field_speaker = compute_field_pressure(field_speaker, speaker);

        % keyboard
        tmp = lens;
        tmp.locs(:, 1) = target.locs(round(end / 2), 1);
        lens.weights_in = getCoordinates(field_speaker, tmp);
        % lens.weights_in = lens.weights_in / sum(abs(lens.weights_in));
        lens.weights_out = lens.weights_in .* lens.delay;
        field_lens = compute_field_pressure(field_lens, lens);

        %% Visulization
        % figure_init('GENERATED SOUND FIELD')
        figure(1)
        set(gcf, 'Position', [210.6000 69.8000 1.0934e+03 980.2000])
        clf
        subplot(221)
        imagesc(field_speaker.y, field_speaker.x, log(abs(field_speaker.sound_pressure)))
        hold on
        plot(target.locs(:, 2), target.locs(:, 1), 'ro')
        plot(speaker.locs(:, 2), speaker.locs(:, 1), 'r*')
        plot(lens.locs(:, 2), field_speaker.x(end), 'gs', 'MarkerFaceColor', 'g')
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
        pbaspect([1, 1, 1])

        % title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
        set(gca, 'fontsize', 20)

        % figure_init('COMPARE RESULTS ANGLE')
        subplot(224)
        plot(angle(target.sound_pressure), 'linewidth', 2)
        hold on
        plot(angle(out), 'linewidth', 2)
        legend('Desired', 'Estimated', 'box', 'off', 'location', 'northeast', 'fontsize', 20)
        pbaspect([1, 1, 1])

        % title(sprintf('Desired angle: %.2f, Real angle: %.2f', target_angle, real_angle))
        set(gca, 'fontsize', 20)

        % figure_init('LENS SOUND FIELD')
        subplot(223)
        imagesc(field_lens.y, field_lens.x, log(abs(field_lens.sound_pressure)))
        hold on
        plot(lens.focusing_point(:, 2), lens.focusing_point(:, 1), 'ro')
        plot(lens.locs(:, 2), lens.locs(:, 1), 'gs', 'MarkerFaceColor', 'g')
        set(gca, 'XTick', [], 'YTick', [])
        title(sprintf('Steering Angle: %.2f', steering_angle), 'FontSize', 20)
        xlabel('y')
        ylabel('x')
        pbaspect([field_lens.len_y, field_lens.len_x, 1])

        % [x, y] = getCoordinatesCore(field_lens, lens.focusing_point);
        % focusing_point_amp = field_lens.sound_pressure(x, y);
        amps(zi) = abs(sum(lens.weights_out));

        [steerVec, theta] = getSteeringMatrix(lens);
        beampattern = lens.weights_out.' * steerVec;
        % theta = theta - 90;
        % beampattern = circshift(beampattern, round(-length(beampattern) / 2));
        figure(3)
        % clf
        plot(theta, (abs(beampattern)), 'linewidth', 2, 'color', cmap(zi, :));
        id = theta == steering_angle;
        amps_record(zi) = abs(beampattern(id));
        hold on
        xlabel('Angle')
        ylabel('Magnitude')
        set(gca, 'fontsize', 20)

        fprintf("**********************\n")
        fprintf("**********************\n")
        fprintf("Amplitude = %.2f in frequency = %i\n", amps(zi), fc)
        fprintf("**********************\n")
        fprintf("**********************\n")

    end
    amps_total(mi) = sum(amps_record);
    fig = figure(3);
    title(sprintf("Energy sum = %.2f", amps_total(mi)))
    filename = "figures2/wide_angle_random_phase_" + string(mi);
    saveas(fig, "./src/" + filename + ".png");
    writematrix(angle(lens_delay), "./src/" + filename + ".txt")

end

%%
figure(4)

clf
histogram(amps_total, 20);
amax = max(amps_total);
amin = min(amps_total);
amedian = median(amps_total);

title(sprintf("max=%.2f, min=%.2f, median=%.2f", amax, amin, amedian))
set(gca, 'fontsize', 20)
fig = figure(4);
filename = "figures2/final_results";
saveas(fig, "./src/" + filename + ".png");
writematrix(amps_total, "./src/" + filename + ".txt")


figure(5)
clf
plot(unwrap(angle(lens.delay)), 'linewidth', 2)
title("lens phase delay")
xlabel("Lens index")
ylabel("Phase delay")
set(gca, 'fontsize', 20)

figure(6)
clf
imagesc(abs(speaker_w))
xlabel("Array")
ylabel("Direction")
title("Amplitude of w")
set(gca, 'fontsize', 20)

% locs = zeros(length(field_lens.x), 2);
% locs(:, 1) = field_lens.x;
% [x, y] = getCoordinatesCore(field_lens, locs);
% aa = field_lens.sound_pressure(x, y);
% figure(3)
% clf
% plot(abs(aa))
