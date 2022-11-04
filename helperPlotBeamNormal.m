function [rx1] = helperPlotBeamNormal(params, lens_locs2D, weights, elevation_theta, observe_axis)
    elevation_theta = elevation_theta / 180 * pi;

    % assert(elevation_theta <= 2 * pi)
    % observe_axis = 'x-axis';

    r = 1e6;
    lens_num = size(lens_locs2D, 1);

    % dmat = zeros(181, 181);
    sample_points = length(elevation_theta);
    rx1 = zeros(sample_points, 1);
    observe_line = zeros(sample_points, 3);

    % keyboard


    for mi = 1 : sample_points
        if(strcmp(observe_axis, 'x-axis'))
            observe_line(mi, 1) = r * cos(elevation_theta(mi));
            observe_line(mi, 2) = 0;
        else
            if(strcmp(observe_axis, 'y-axis'))
                observe_line(mi, 1) = 0;
                observe_line(mi, 2) = r * cos(elevation_theta(mi));
            else
                error("wrong target axis!")
            end
        end
        observe_line(mi,3) = r * sin(elevation_theta(mi));
    end
    % observe_line(:,3) = r * sin(elevation_theta(1)/180*pi);

    for mi = 1 : size(observe_line, 1)
        target = observe_line(mi, :);
        for ti = 1:lens_num
            d = compute_distance(target, lens_locs2D(ti, :));
            rx1(mi) = rx1(mi) + weights(ti) * exp(-1j * 2 * pi * params.fc / params.c * d) / d;
        end
    end
end