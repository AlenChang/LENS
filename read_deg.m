function target = read_deg(data, target_direction)

    ang = 0:5:90;


    
    x = data(:, 1);
    y = data(:, 2);
    z = data(:, 3);
    data(:, 1:3) = [];

    a = data(:,target_direction == ang);
    
    xnew = linspace(min(x), max(x), 80);
    % znew = linspace(min(z), max(z), 41);
    tnew = linspace(0, sqrt(max(z)^2+max(y)^2), 80);
    b = sqrt(max(z)^2+max(y)^2);
    t = ceil(sqrt(max(abs(y)).^2 + max(z).^2));
    
    target = zeros(41, 41);
    for ti = 1:length(a)
        aa = abs(a(ti));
        tl = floor((sqrt(y(ti)^2+z(ti)^2)+1)/b*40)+1;
        xl = floor((x(ti) + 1000 + 1) / 2000 * 40)+1;
        target(tl, xl) = target(tl, xl) + aa;
    end

    target = flipud(target);
    target(1:2, :) = [];
    target(end-2:end, :) = [];

end