function d = compute_distance(x, y)
assert(numel(x) == numel(y) && numel(y) >= 2)
x = manipulate_dimention(x);
y = manipulate_dimention(y);
d = x - y;
% keyboard
d = norm(d);
end

function A = manipulate_dimention(A)
A = reshape(A, 1, []);
end