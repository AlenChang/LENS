function [speaker, R] = SRF_solution(target, speaker)

    R = transfer_func(target, speaker);
    speaker.weights_out = solve_least_square(R, target.sound_pressure);
    % keyboard
    speaker.weights_out = speaker.weights_out ./ max(abs(speaker.weights_out));

    fprintf("================\n")
    fprintf("Amplitude of computed weights:\n")

    for ti = 1:speaker.numel
        fprintf("|weights(%i)| = %.3f, \angle weights(%i) = %.3f \n", ...
            ti, abs(speaker.weights_out(ti)), ti, angle(speaker.weights_out(ti)) / pi * 180)
    end

end

function R = transfer_func(target, speaker)
    R = zeros(target.numel, speaker.numel);
    % keyboard
    d = zeros(target.numel, speaker.numel);
    for ti = 1:target.numel
        for ni = 1:speaker.numel
            d(ti, ni) = compute_distance(target.locs(ti, :), speaker.locs(ni, :));
        end
    end
    R = exp(-1j * 2 * pi * speaker.fc * d / speaker.c) ./ (2 * pi * d);
end

function weights = solve_least_square(R, b)
    % keyboard
    weights = pinv(R' * R) * R' * b / (b' * b);
end
