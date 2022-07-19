function field = compute_field_pressure(field, speaker)
fprintf("================\n")
fprintf("Computing sound presure fiel with given speakers...\n")
for ti = 1:speaker.num
    locs = speaker.locs(ti, :);
    for mi = 1:size(field.locs, 1)
        for ni = 1:size(field.locs, 2)
            d = compute_distance(field.locs(mi, ni, :), locs);
            field.sound_pressure(mi, ni) = field.sound_pressure(mi, ni) +...
                speaker.weights(ti) * exp(1j * (-2 * pi * speaker.fc * d / speaker.c));
        end

    end
end
fprintf("Done!!!\n")

end