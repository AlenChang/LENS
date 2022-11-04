
speaker_num = 2:2:16;
out_amps = zeros(length(speaker_num), 181);
counter = 1;

phased_array_only = zeros(size(speaker_num));
for ti = speaker_num
    opt_file = strcat("vary_speaker_num/optimal", string(ti), ".mat");
    para_file = strcat("vary_speaker_num/parameters", string(ti), ".mat");

    load(opt_file)
    load(para_file)
    powers = sum(abs(speaker_w).^2, 2);
    speaker_w = speaker_w ./ sqrt(powers);
    lens_out = speaker_w * G * diag(len_theta);
    beams = lens_out * steerVec;

    out_amps(counter, :) = diag(beams);
    
    % keyboard

    % phased array only
    % num_speaker = size(G, 1);
    speaker_locs = generate1Dlocs(params, ti);
    speaker_weights = getPhasedArraySteeringWeights(params, ti);
    speaker_weights = speaker_weights ./ sqrt(sum(abs(speaker_weights).^2, 2));
    speaker_out = speaker_weights * steerVec(1:ti,:);
    phased_array_only(counter) = speaker_out(91, 91);

    counter = counter + 1;
end

data = (abs(out_amps(:, 31:91))').^2;
data(:,2) = data(:,2)-1.5;
data = data;
data = data / max(data(:));

vary_speaker_num.x = x;
vary_speaker_num.y = data;
vary_speaker_num.label = {"m = 2", "m = 4", "m = 6", "m = 8", "m = 10", "m = 12", "m = 14", "m = 16"};

% colors = colorcube(length(speaker_num)+1);
figure(1)
clf
x = 30:90;
hold on
% colormap(colors)
% for ti = 1 : length(speaker_num)
plot(x, data(:, 1:7), 'linewidth', 3)
hold on
plot(x, data(:, 8), 'k', 'linewidth', 3)
% end
set(gca, 'fontsize', 20, 'linewidth', 2, 'ytick', [0:0.2:1], 'box', 'on', 'xticklabel', [-60:10:0])
axis([30, 90, 0, 1.05])
xlabel("Angle (Azimuth, \circ)", 'fontsize', 30)
ylabel("Normalized Power", 'fontsize', 30)
pbaspect([6,3,1])
legend("m = 2", "m = 4", "m = 6", "m = 8", "m = 10", "m = 12", "m = 14", "m = 16",'location', 'northoutside', 'numcolumns', 4, 'box', 'off')

print('-depsc', 'fig_results/vary_speaker_num.eps')
saveas(gcf, 'figs/matlab_test.png')

phased_array_only = phased_array_only / max(abs(phased_array_only));

coeff = abs(phased_array_only(3)).^2 / data(end, 3).^2;
% figure(2)
% clf
% plot(speaker_num, abs(phased_array_only).^2 / coeff / 3, 'linewidth', 2)
% hold on
% plot(speaker_num, data(end, :).^2, 'linewidth', 2)
% saveas(gcf, 'figs/matlab_test.png')