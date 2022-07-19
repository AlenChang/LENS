function target = desired_pressure_at_samples(target, speaker)
fprintf("================\n")
if(strcmp(target.coordinates, 'circle'))
    target.desired_angle = (rand(1) * 2 / 3 + 1 / 6) * pi;
    target.desired_phase = rand(1) * 2 * pi;
    xshift = (target.desired_angle - pi / 2) / pi * speaker.num;
    x = linspace(-speaker.num/2, speaker.num/2, target.num) - xshift;
    
    
    target.sound_pressure = abs(sinc(x))' .* exp(1j*target.desired_phase);
    fprintf("Target sound field: target.desired_angle = %.2f degree\n", target.desired_angle / pi *180);
    fprintf("Target sound field: Initial Phase = %.2f degree\n", target.desired_phase / pi * 180);
else
    error("DESIRED PRESSURE: target_coordinates type not defined!\n")
end


end