function lens = backstepping(lens, focusing_point)
lens.focusing_point = focusing_point;
lens.weights = zeros(lens.num, 1);
for ti = 1 : lens.num
    d = compute_distance(lens.locs(ti, :), lens.focusing_point);
    lens.weights(ti) = conj(exp(-1j*2*pi*lens.fc/lens.c*d));
end
lens.weights = lens.weights .* kaiser(lens.num, 2);
end