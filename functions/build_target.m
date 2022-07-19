function target = build_target(num_target, target_coordinates)
fprintf("================\n")
target.num =num_target;
target.locs = zeros(target.num, 2);
target.coordinates = target_coordinates;


if(strcmp(target.coordinates , "circle"))
    target.phi = linspace(0, pi, target.num);
    target.R = 6 * ones(target.num, 1);
    for ti = 1 : target.num
        target.locs(ti, 1) = target.R(ti) * sin(target.phi(ti));
        target.locs(ti, 2) = target.R(ti) * cos(target.phi(ti));
    end
else
    error("BUILD TARGET: target_coordinates type not defined!\n")
end

fprintf("Target config:\n")
fprintf("        Number of Targets: %i\n", target.num)
fprintf("        Target geometry: %s\n", target.coordinates)

end