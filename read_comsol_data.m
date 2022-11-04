function aa = read_comsol_data(filename)
    data = readtable(filename);

    for ti = height(data):-1:1
        if(contains(data{ti, 1}, "%"))
            data(ti,:) = [];
        end
    end
    aa = zeros(size(data));
    for ti = 1 : size(aa, 1)
        for mi = 1 : size(aa, 2)
            tmp = data{ti, mi};
            if(iscell(tmp))
                aa(ti, mi) = str2num(tmp{1});
            else
                aa(ti, mi) = tmp;
            end
            
        end
    end
end