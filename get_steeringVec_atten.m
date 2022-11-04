lens_num = 16;
target_angle = 90;


c = 34300;
fc = 20e3;
lambda = c / fc;
lens_locs = (- (lens_num - 1):2:lens_num - 1) * lambda / 4;
lens_locs = [lens_locs' zeros(lens_num, 1)];

theta = (0:180) / 180 * pi;


for zi = 1:length(theta)
    out = 0;
    target_locs = [r * cos(theta(zi)), r * sin(theta(zi))];

    for mi = 1:lens_num
        d = compute_distance(target_locs, lens_locs(mi, :));
        out = out + lens_out(target_angle+1, mi) * exp(-1j * 2 * pi * fc / c * d) / (2 * pi * d);
    end

    bp(ti, zi) = out;
end
