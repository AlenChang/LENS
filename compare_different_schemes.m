swol = load('comsol_data/compare/single_speaker_wo_lens.txt');
swol_ang = atan(swol(:, 3) ./ swol(:, 2)) / pi * 180;



[swol, swol_ang] = load_comsol_compare_data('comsol_data/compare/single_speaker_wo_lens.txt');
[swl, swl_ang] = load_comsol_compare_data('comsol_data/compare/single_speaker_w_lens.txt');
[mwol, mwol_ang] = load_comsol_compare_data('comsol_data/compare/multi_speaker_wo_lens.txt');
[mwl, mwl_ang] = load_comsol_compare_data('comsol_data/compare/multi_speaker_w_lens.txt');

figure(1)
clf
hold on
h1 = plot(swol_ang, swol, 'linewidth', 3);
h2 = plot(swl_ang, swl, 'linewidth', 3);
h3 = plot(mwol_ang, mwol, 'linewidth', 3);
h4 = plot(mwl_ang, mwl, 'linewidth', 3);

set(gca, 'linewidth', 2, 'box', 'on', 'fontsize', 20)
xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
ylabel('Magnitude', 'fontsize', 30)
axis([-90, 90, 0, 13])
set(gca, 'xtick', -90:30:90)
pbaspect([6,3,1])
legend([h1, h3, h2, h4], 'Point Source', 'Phased Array', 'Point Source + Lens', 'Phased Array + Lens', 'fontsize', 20, 'location', 'north', 'box', 'off', 'numcolumns', 2)
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/compare_schemes.eps')
