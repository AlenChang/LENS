function [rx1] = helperPlotFarfield(params, lens_locs2D, weights, elevation_theta)
    r = 1e6;
    lens_num = size(lens_locs2D, 1);

    % dmat = zeros(181, 181);
    sample_points = length(elevation_theta);
    rx1 = zeros(sample_points^2, 1);
    observe_plane = zeros(sample_points^2, 3);
    square_len = r * cos(elevation_theta(1)/180*pi);

    sample_x = linspace(-square_len, square_len, sample_points);
    for ti = 1 : sample_points
        for mi = 1 : sample_points
            observe_plane((ti-1)*sample_points+mi, 1) = sample_x(ti);
            observe_plane((ti-1)*sample_points+mi, 2) = sample_x(mi);
            % observe_plane(:,3) = sqrt(r^2 - sample_x(ti)^2 - sample_x(mi)^2);
            observe_plane(:,3) = r * sin(elevation_theta(1)/180*pi);
        end
    end
    % observe_plane(:,3) = r * sin(elevation_theta(1)/180*pi);

    for mi = 1 : size(observe_plane, 1)
        target = observe_plane(mi, :);
        for ti = 1:lens_num
            d = compute_distance(target, lens_locs2D(ti, :));
            rx1(mi) = rx1(mi) + weights(ti) * exp(-1j * 2 * pi * params.fc / params.c * d) / d;
        end
    end
    rx1 = reshape(rx1, sample_points, sample_points);
end