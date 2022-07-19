function target = desired_pressure_at_samples_lens(target, lens)
target.sound_pressure = zeros(target.num, 1);
if(strcmp(target.coordinates, 'square'))
    x_interp = target.locs(target.y_start : target.y_end, 2);
    y_interp = lens.locs(:, 2);
    y = interp1(y_interp, lens.weights, x_interp, 'linear');
    y(isnan(y)) = 0;
    filters = ones(10, 1) / 10;
    y = conv(y, filters, 'same');
    target.sound_pressure(target.y_start : target.y_end) = y;
elseif(strcmp(target.coordinates, 'raw'))
    target.sound_pressure(target.y_start : target.y_end) = lens.weights;
else
    error("SOUND PRESSURE: target_coordinates type not defined!\n")
end

end