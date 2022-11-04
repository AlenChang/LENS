clear 

% target_plane = 'yz2';
% filename = strcat('comsol_data/6_speaker/6_speaker_line_all.txt');
filename = strcat('comsol_data/6_speaker_line_200.txt');
bp = read_comsol_data(filename);

bpx = bp(:, 1:2);
bpx(:, 2) = bpx(:, 2) - min(bpx(:, 2));
bpangle = unwrap(acos(bpx(:, 1) ./ sqrt(bpx(:, 1).^2 + bpx(:, 2).^2)));
a1 = linspace(pi, 0, 181);

bp = bp(:, 3:end);
ang_width = size(bp, 2);
bpnew = zeros(181, ang_width);

for ti = 1:ang_width
    bpnew(:, ti) = interp1(bpangle, bp(:, ti), a1, 'linear');
end

save_data = bpnew((end+1)/2, :);

bpnew = bpnew(31-7:151+7, :);

bp = bpnew;
bp = (abs(bp).^2)';

aa = bp / 100;
aa = aa / max(aa(:));
figure(1)
clf
imagesc(aa)
hold on
% plot(1:181, 1:181, 'r*')
set(gca, 'xtick', linspace(1, size(bpnew, 1), 5), 'ytick', linspace(1, size(bpnew, 2), 5))
set(gca, 'xticklabel', -60:30:60, 'yticklabel', -60:30:60)
set(gca, 'fontsize', 20)
xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
ylabel('Steering Angle (\circ)', 'fontsize', 30)
colorbar('Ticks', 0:0.2:1)
pbaspect([1, 1, 1])
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/beam_pattern_comsol.eps')



function data = read_comsol_data(filename)
    fclose all
    Counter = 1;
    FID = fopen(filename, 'rt');
    tline = fgetl(FID);
    File_Data{Counter} = tline;

    while ischar(tline)
        Counter = Counter + 1;
        tline = fgetl(FID);
        File_Data{Counter} = tline;
    end

    File_Data = File_Data(10:end);

    min_len = length(split(File_Data{12}));
    data = zeros(length(File_Data) - 1, min_len);

    for ti = 1:length(File_Data) - 1
        tmp = File_Data{ti};
        tmp = split(File_Data{ti});

        for ni = 1:min_len
            tmp2 = tmp{ni};
            data(ti, ni) = str2num(tmp2);
        end

    end

end
