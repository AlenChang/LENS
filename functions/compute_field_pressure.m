function field = compute_field_pressure(field, speaker)
    fprintf("================\n")
    fprintf("Computing sound presure fiel with given speakers...\n")
    field_locs = field.locs;

    for ti = 1:speaker.num
        locs = speaker.locs(ti, :);
        d = zeros(size(field.sound_pressure));

        for mi = 1:size(field_locs, 1)

            for ni = 1:size(field_locs, 2)
                d(mi, ni) = compute_distance(field_locs(mi, ni, :), locs);
            end

        end

        field.sound_pressure = field.sound_pressure + ...
            speaker.weights_out(ti) * exp(1j * (-2 * pi * speaker.fc * d / speaker.c)) ./ (2 * pi * d);
        %                 ./ (2*pi*sqrt(d));
    end

    % fprintf("Done!!!\n")

end
