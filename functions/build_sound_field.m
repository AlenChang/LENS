function field = build_sound_field(field_len, gridsize)
field.gridsize = gridsize;
field.len = field_len / 2;
field.x = -field.len:gridsize:field.len;
field.y = -field.len:gridsize:field.len;

field.sound_pressure = zeros(length(field.x), length(field.y));

field.locs = zeros(length(field.x), length(field.y), 2);
for ti = 1:length(field.x)
    for mi = 1:length(field.y)
        field.locs(ti, mi, 1) = field.x(ti);
        field.locs(ti, mi, 2) = field.y(mi);
    end
end
fprintf("================\n") 
fprintf('Sound field size: [-%i cm, %i cm]\n', field.len, field.len)
fprintf('Sound field grid size: %i * %i\n', size(field.sound_pressure, 1), size(field.sound_pressure, 1))
end