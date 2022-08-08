function target = build_target_lens(field, target_coordinates, lens)
fprintf("================\n")
target.locs = [];
target.coordinates = target_coordinates;
len_x = size(field.locs, 1);
% keyboard
if(strcmp(target.coordinates , "square"))
    locs = field.locs(ceil(len_x / 2) : len_x, 1, :);
    target = append_locs(target, locs);

    target.y_start = size(locs, 1) + 1;
    
    locs = field.locs(end, 2:end-1, :);
    target = append_locs(target, locs);

    target.y_end = target.y_start + size(locs, 2) - 1;

    locs = field.locs(end:-1:ceil(len_x / 2), end, :);
    target = append_locs(target, locs);
elseif(strcmp(target.coordinates , "raw"))
    % keyboard
    locs = field.locs(ceil(len_x / 2) : len_x, 1, :);
    target = append_locs(target, locs);

    target.y_start = size(locs, 1) + 1;
    
    x = field.locs(end, 1);
    locs = lens.locs;
    locs(:, 1) = x;
    % keyboard
    target = append_locs(target, locs);

    target.y_end = target.y_start + size(locs, 1) - 1;

    locs = field.locs(end:-1:ceil(len_x / 2), end, :);
    target = append_locs(target, locs);

    
else
    error("BUILD TARGET: target_coordinates type not defined!\n")
end

target.num = size(target.locs, 1);

fprintf("Target config:\n")
fprintf("        Number of Targets: %i\n", target.num)
fprintf("        Target geometry: %s\n", target.coordinates)

end

function target = append_locs(target, locs)
locs = squeeze(locs);
target.locs = [target.locs; locs];
end