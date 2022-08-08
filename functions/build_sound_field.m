function field = build_sound_field(field_len_x, field_len_y, gridsize)
    % consider the 2D field image
    field.gridsize = gridsize;
    field.len_x = field_len_x / 2;
    field.len_y = field_len_y / 2;
    field.x = -field.len_x:gridsize:field.len_x;
    field.y = -field.len_y:gridsize:field.len_y;

    field.sound_pressure = zeros(length(field.x), length(field.y));

    field.locs = zeros(length(field.x), length(field.y), 3);

    for ti = 1:length(field.x)

        for mi = 1:length(field.y)
            field.locs(ti, mi, 1) = field.x(ti);
            field.locs(ti, mi, 2) = field.y(mi);
        end

    end

    fprintf("================\n")
    fprintf('Sound field size: [-%i cm, %i cm]\n', field.len_x, field.len_y)
    fprintf('Sound field grid size: %i * %i\n', size(field.sound_pressure, 1), size(field.sound_pressure, 1))
end
