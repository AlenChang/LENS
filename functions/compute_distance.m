function d = compute_distance(x, y)
assert(numel(x) == 2 && numel(y) == 2)
x = manipulate_dimention(x);
y = manipulate_dimention(y);
d = x - y;
d = sqrt(d(1)^2 + d(2)^2);
end

function A = manipulate_dimention(A)
A = reshape(A, 1, []);
end