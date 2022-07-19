function out = getCoordinates(field, target)
out = zeros(size(target.locs, 1), 1);
for ti = 1:target.num
    x = round((target.locs(ti, 1) - field.x(1)) / field.gridsize + 1);
    y = round((target.locs(ti, 2) - field.y(1)) / field.gridsize + 1);
    out(ti) = field.sound_pressure(x, y);
end
end