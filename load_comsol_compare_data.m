function [data, data_ang] = load_comsol_compare_data(filename)
    data = load(filename);
    data_ang = atan(data(:, 2) ./ data(:, 3)) / pi * 180;
    data = data(:, 4);
end