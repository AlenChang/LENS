function speaker = build_speakers(num_speakers, speaker_center, speaker_spacing, fc, field_type)
    speaker.design_fc = 20e3;
    speaker.c = 34300;
    speaker.fc = fc;
    speaker.lambda = speaker.c / speaker.fc;
    speaker.speaker_spacing = speaker.c / speaker.design_fc * speaker_spacing;
    speaker.field_type = field_type;
    speaker.num = num_speakers;

    switch (field_type)
        case '1D'
            speaker.numel = speaker.num;
            speaker = generate_specs(speaker, speaker_center);

            for ti = 1:speaker.num
                speaker.locs(ti, 2) = (- (speaker.num - 1) / 2 + ti - 1) * speaker.speaker_spacing;
            end

            speaker.locs = speaker.locs + speaker.center;

            fprintf("================\n")
            fprintf("Speaker config:\n")
            fprintf("        Number of Speakers: %i\n", speaker.num)
            fprintf("        Central frequency: %i kHz\n", speaker.fc / 1000)
            fprintf("        Wave length: %f cm\n", speaker.lambda)
        case '2D'
            speaker.numel = speaker.num^2;
            speaker = generate_specs(speaker, speaker_center);

            % fixed in x dimension, we only update locs in y and z axis
            % for each row in z axis (along y axis)
            index = zeros(speaker.num / 2, speaker.num / 2);
            counter = 0
            for mi = 1:speaker.num
                z_locs = (- (speaker.num - 1) / 2 + mi - 1) * speaker.speaker_spacing;
                % for each row in y axis (along z axis)
                for ti = 1:speaker.num
                    y_locs = (- (speaker.num - 1) / 2 + ti - 1) * speaker.speaker_spacing;
                    counter = (mi - 1) * speaker.num + ti;
                    % index(mi, ti) = counter;
                    speaker.locs(counter, 2:3) = [y_locs, z_locs];
                    
                end

            end

            
            % mat_right_up = fliplr(index);
            % orders = reshape(mat_right_up', [], 1);
            % speaker.locs(counter+1:2*counter, 2:3) = speaker.locs(orders, 2:3)
            % speaker.locs(end/2+1:end, 2:3) = flipud(speaker.locs(1:end/2, 2:3));

            % keyboard
            speaker.locs = speaker.locs + speaker.center;

            fprintf("================\n")
            fprintf("Speaker config:\n")
            fprintf("        Number of Speakers: %i\n", speaker.numel)
            fprintf("        Central frequency: %i kHz\n", speaker.fc / 1000)
            fprintf("        Wave length: %f cm\n", speaker.lambda)

            % keyboard

        otherwise
            error('Wrong field type. Muse be 1D or 2D.')
    end

    PLOT = true;

    if (PLOT)
        figure(11)
        clf
        plot3(speaker.locs(:, 1), speaker.locs(:, 2), speaker.locs(:, 3), '*')
        pbaspect([1, 1, 1])
        saveas(gcf, 'figs/2D_speaker.png')
    end

end

function speaker = generate_specs(speaker, speaker_center)
    speaker.center = zeros(1, 3);
    speaker.center(1:2) = speaker_center;
    speaker.locs = zeros(speaker.numel, 3);
    speaker.weights_out = ones(speaker.numel, 1);
end
