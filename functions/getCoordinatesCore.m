function [x, y] = getCoordinatesCore(field, locs)
assert(size(locs, 2) == 2)
num = size(locs, 1);
x = zeros(num, 1);
y = zeros(num, 1);
for ti = 1 : num
    x(ti) = round((locs(ti, 1) - field.x(1)) / field.gridsize + 1);
    y(ti) = round((locs(ti, 2) - field.y(1)) / field.gridsize + 1);
end
end