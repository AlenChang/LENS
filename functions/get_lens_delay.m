function lens = get_lens_delay(lens, speaker, field)
lens_locs = lens.locs;
lens_locs(:, 1) = squeeze(field.locs(end, 1, 1));
lens.delay = zeros(lens.num, 1);
for ti = 1 : lens.num
    d = compute_distance(speaker.center, lens_locs(ti, :));
    lens.delay(ti) = exp(1j* 2 * pi * speaker.design_fc * d / speaker.c);
end
lens.back_step = lens.delay;

end