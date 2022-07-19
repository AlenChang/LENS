function out = plot_max_value(x, data)
[~, idx1] = max(abs(data));
stem(x(idx1), abs(data(idx1)), 'linewidth', 2)
out = x(idx1);
end