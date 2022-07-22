function lens = backstepping(lens, focusing_point, focusing_type)
lens.weights = zeros(lens.num, 1);
lens.focusing_type = focusing_type;
lens.focusing_point = focusing_point;

if(strcmp(lens.focusing_type, 'point'))
    
    for ti = 1 : lens.num
        d = compute_distance(lens.locs(ti, :), lens.focusing_point);
        lens.weights(ti) = conj(exp(-1j*2*pi*lens.fc/lens.c*d));
    end
elseif(strcmp(lens.focusing_type, 'direction'))
    steervec = focusing_point - lens.center;
    theta = atan(steervec(2) / steervec(1));
    for ti = 1 :lens.num
        if(ti == 1)
            lens.weights(ti) = 1;
        else
            d = compute_distance(lens.locs(ti,:), lens.locs(1,:));
            d = d * sin(theta);
            lens.weights(ti) = exp(-1j*2*pi*lens.fc/lens.c*d) * conj(lens.back_step(ti));
        end
    end
else
    error("BACKSTEPPING: focusing type not defined!\n")
end
% lens.weights = lens.weights .* kaiser(lens.num, 2);
end