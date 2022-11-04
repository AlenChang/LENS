function speaker_locs = generate1Dlocs(params, num_source)
    speaker_locs = zeros(num_source, 3);
    speaker_locs(:, 1) = (- (num_source - 1):2:num_source - 1) * params.lambda / 4;

end
