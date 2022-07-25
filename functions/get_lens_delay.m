function lens = get_lens_delay(lens, speaker, field)
    lens_locs = lens.locs;
    lens_locs(:, 1) = squeeze(field.locs(end, 1, 1));
    lens.delay = zeros(lens.num, 1);

    for ti = 1:lens.num
        d = compute_distance(speaker.center, lens_locs(ti, :));
        delay = get_phase_delay(lens, d);
        delay1 = exp(1j * 2 * pi * lens.design_fc * d / lens.c);
        % p1 = angle(delay1);
        % if(p1 < 0)
        %     p1 = p1 + 2 * pi;
        % end
        fprintf("delay = %.2f, delay1 = %.2f\n", angle(delay), angle(delay1))
        lens.delay(ti) = delay;
    end

    lens.back_step = lens.delay;

end

function Leff = get_LEFF(speaker)
    fc = speaker.design_fc;
    c = speaker.c;
    lambda = c / fc;
    Leff = ((0:speaker.num - 1)' / speaker.num + 1) * lambda;
end

function Leff_sel = select_LEFF(speaker, d)
    delay_array = (0:2 * pi / speaker.num:2 * pi)';
    Leff = get_LEFF(speaker);
    delta_d = mod(d, speaker.c / speaker.design_fc);
    target_delay = 2 * pi * speaker.design_fc * delta_d / speaker.c;
    delta_phase = target_delay - delay_array;
    [~, id] = min(abs(delta_phase));

    if (id == length(delay_array))
        id = 1;
    end

    Leff_sel = Leff(id);
end

function delay = get_phase_delay(speaker, d)
    Leff_sel = select_LEFF(speaker, d);
    lambda = speaker.c / speaker.fc;
    delay = exp(1j * (Leff_sel / lambda - 1) * 2 * pi);

end
