
lens_nums = [4, 8, 16, 24, 32];
out_amps = zeros(length(lens_nums), 181);
counter = 1;
for ti = lens_nums
    opt_file = strcat("vary_lens_size/optimal", string(ti), ".mat");
    para_file = strcat("vary_lens_size/parameters", string(ti), ".mat");

    load(opt_file)
    load(para_file)

    lens_out = speaker_w * G * diag(len_theta);
    beams = lens_out * steerVec;

    out_amps(counter, :) = diag(beams);
    counter = counter + 1;
end

data = (abs(out_amps(:, 31:91))').^2;
data = data / max(data(:));

vary_lens_size.x = x;
vary_lens_size.y = data;
vary_lens_size.label = {"4\times4", "8\times8", "16\times16", "24\times24", "32\times32"};

figure(1)
clf
x = 30:90;
plot(x, data, 'linewidth', 3)
set(gca, 'fontsize', 20, 'linewidth', 2, 'ytick', [0:0.2:1], 'xticklabel', [-60:10:0])
axis([30, 90, 0, 1.05])
xlabel("Angle (Azimuth, \circ)", 'fontsize', 30)
ylabel("Normalized Power", 'fontsize', 30)
pbaspect([6,3,1])
legend("4\times4", "8\times8", "16\times16", "24\times24", "32\times32", 'location', 'northoutside', 'numcolumns', 5, 'box', 'off')

print('-depsc', 'fig_results/vary_lens_size.eps')
saveas(gcf, 'figs/matlab_test.png')