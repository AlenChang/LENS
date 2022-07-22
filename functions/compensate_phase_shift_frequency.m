function lens = compensate_phase_shift_frequency(lens)
d = abs(lens.center(1));
L = abs(lens.locs(end, 2));
k = (d - sqrt(d^2 - L^2)) / L^2;

x = lens.locs(:, 1);

phase_comp = k / lens.c * 2 * pi * (lens.fc - lens.design_fc) * x.^2 ...
    + d / lens.c * 2 * pi * (lens.fc - lens.design_fc);
figure(11)
clf
hold on
plot(unwrap(angle(lens.delay)), 'LineWidth',  2)
lens.back_step = lens.delay .* exp(1j * phase_comp);
plot(unwrap(angle(lens.back_step)), 'LineWidth',  2)
% pause
end