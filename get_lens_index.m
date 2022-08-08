function lens_index = get_lens_index(len_theta)
    lens_angle = angle(len_theta);
    thre = 2 * pi / 16;
    % thre = 0;
    lens_angle(lens_angle > thre / 2) = lens_angle(lens_angle > thre / 2) - 2 * pi;
    lens_index = zeros(size(lens_angle));
    for ti = 1 : length(lens_angle)
        lens_index(ti) = specify_lens_index(lens_angle(ti));
    end
    lens_index = lens_index - 1;
end