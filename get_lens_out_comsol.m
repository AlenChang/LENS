function out = get_lens_out_comsol(data)
    row = sum(data);
    % keyboard
    data(:, isnan(row)) = [];
    
    x = data(:, 1);
    lambda = 34300 / 20e3 / 2;
    xtarget = (-15:2:15) * lambda / 2 * 10;
    data = data(:, 3:end);
    % out = zeros(16, 181);
    index = zeros(16, 1);
    for ti = 1 : length(xtarget)
        tmp = x - xtarget(ti);
        % xtarget(ti)
        [~, index(ti)] = min(abs(tmp));
        % x(index(ti))
    end
    out = data(index, :);
    % out = out(:, end-180:end);
    % assert(size(out, 1) == 16)
    % index
end