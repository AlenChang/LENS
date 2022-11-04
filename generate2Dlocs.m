function lens_locs = generate2Dlocs(params, num_source)

    % lens_locs = (- (num_source - 1):2:num_source - 1) * params.lambda / 4;
    % lens_locs = [lens_locs' zeros(num_source, 1)];
    
    lens_locs_x = (- (sqrt(num_source) - 1):2:sqrt(num_source) - 1) * params.lambda / 4;
    
    len_1D = length(lens_locs_x);
    lens_locs = zeros(len_1D^2, 3);
    for ti = 1:len_1D
        for mi = 1:len_1D
            lens_locs((ti - 1) * len_1D + mi, 2) = lens_locs_x(ti);
            lens_locs((ti - 1) * len_1D + mi, 1) = lens_locs_x(mi); % array in x axis
            lens_locs((ti - 1) * len_1D + mi, 3) = 0;
        end
    
    end
end