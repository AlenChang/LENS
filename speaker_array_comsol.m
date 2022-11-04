
clear

target_plane = 'yz2';
filename = strcat('comsol_data/6_speaker_only_yz.txt');

% data = read_comsol_data(filename);
data = readtable(filename);
data = table2array(data);

target_direction = 30;
im = process_comsol_plot_data(data, target_direction);

figure(1)
clf
colormap(parula)
imagesc(db(abs(im)))
pbaspect([1,1,1])
set(gca, 'xticklabel', [], 'yticklabel', [])
print('-depsc', 'fig_results/visu_wo_lens30.eps')
saveas(gcf, 'figs/matlab_test.png')

target_direction = 60;
im = process_comsol_plot_data(data, target_direction);

figure(1)
clf
colormap(parula)
imagesc(db(abs(im)))
pbaspect([1,1,1])
set(gca, 'xticklabel', [], 'yticklabel', [])
print('-depsc', 'fig_results/visu_wo_lens60.eps')
saveas(gcf, 'figs/matlab_test.png')

target_direction = 90;
im = process_comsol_plot_data(data, target_direction);
figure(1)
clf
colormap(parula)
imagesc(db(abs(im)))
pbaspect([1,1,1])
set(gca, 'xticklabel', [], 'yticklabel', [])
print('-depsc', 'fig_results/visu_wo_lens90.eps')
saveas(gcf, 'figs/matlab_test.png')



clear

target_plane = 'deg2';
filename = strcat('comsol_data/6_speaker_only_deg.txt');

% data = read_comsol_data(filename);
data = readtable(filename);
data = table2array(data);

target_direction = 30;
% im = process_comsol_plot_data(data, target_direction);
target = read_deg(data, target_direction);
im = interp2(target, 5, 'linear');

figure(1)
clf
colormap(parula)
imagesc(db(abs(im)))
pbaspect([1,1,1])
set(gca, 'xticklabel', [], 'yticklabel', [])
print('-depsc', 'fig_results/visu_wo_lens30_deg.eps')
saveas(gcf, 'figs/matlab_test.png')

target_direction = 60;
% im = process_comsol_plot_data(data, target_direction);
target = read_deg(data, target_direction);
im = interp2(target, 5, 'linear');

figure(1)
clf
colormap(parula)
imagesc(db(abs(im)))
pbaspect([1,1,1])
set(gca, 'xticklabel', [], 'yticklabel', [])
print('-depsc', 'fig_results/visu_wo_lens60_deg.eps')
saveas(gcf, 'figs/matlab_test.png')

target_direction = 90;
% im = process_comsol_plot_data(data, target_direction);
target = read_deg(data, target_direction);
im = interp2(target, 5, 'linear');
figure(1)
clf
colormap(parula)
imagesc(db(abs(im)))
pbaspect([1,1,1])
set(gca, 'xticklabel', [], 'yticklabel', [])
print('-depsc', 'fig_results/visu_wo_lens90_deg.eps')
saveas(gcf, 'figs/matlab_test.png')

