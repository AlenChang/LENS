function speaker = SRF_solution(target, speaker)

R = transfer_func(target, speaker);
speaker.weights = solve_least_square(R, target.sound_pressure);

fprintf("================\n")
fprintf("Amplitude of computed weights:\n")
for ti = 1 : speaker.num
    fprintf("|weights(%i)| = %.3f\n", ti, abs(speaker.weights(ti)))
end

end

function R = transfer_func(target, speaker)
R = zeros(target.num, speaker.num);

for ti = 1:target.num
    for ni = 1:speaker.num
        d = compute_distance(target.locs(ti, :), speaker.locs(ni, :));
        R(ti, ni) = exp(-1j * 2 * pi * speaker.fc * d / speaker.c) ./ (2 * pi * d);
    end
end
end

function weights = solve_least_square(R, b)
weights = pinv(R' * R) * R' * b / (b'*b);
end