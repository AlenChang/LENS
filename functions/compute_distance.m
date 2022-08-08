function d = compute_distance(x, y)
assert(numel(x) == 3 && numel(y) == 3)
x = manipulate_dimention(x);
y = manipulate_dimention(y);
d = x - y;
% keyboard
d = norm(d);
end

function A = manipulate_dimention(A)
A = reshape(A, 1, []);
end