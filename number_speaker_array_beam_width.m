
addpath('./functions/')
% clear

load parameters_paper.mat

params.c = 34300;
params.fc = 20e3;
params.lambda = params.c / params.fc;

nums = [2, 4, 6, 8, 12, 16];



beams_out = zeros(length(nums), 181);
hpbw = @(N) 2*(pi / 2 - acos(1.391*params.lambda / pi ./ N / (params.lambda / 2))) / pi * 180;

for ti = 1:length(nums)
num_speaker = nums(ti);
target_direction = 91;
speaker_locs = generate1Dlocs(params, num_speaker);
speaker_weights = getPhasedArraySteeringWeights(params, num_speaker);
speaker_out = speaker_weights(target_direction, :) * steerVec(1:num_speaker,:);
% plot(-90:90, abs(speaker_out))
beams_out(ti, :) = abs(speaker_out);
disp(sprintf("For N = %i, Beamwidth = %.2f", nums(ti), hpbw(nums(ti))))

end
beams_out = beams_out';
max_beams = max(beams_out);
beams_out = beams_out ./ max_beams;
angs = -90:90;
figure(1)
clf
hold on
h1 = plot(angs, db(beams_out), 'linewidth', 2);
for ti = 1 : length(nums)
    bw = db(beams_out(:, ti));
    bw = sum(bw >= -3);
    disp(sprintf("Speaker number: %i, beam width: %i\n", nums(ti), bw))
end
hold on
h2 = plot(angs, -3*ones(size(angs)), 'r--', 'linewidth', 3);

set(gca, 'linewidth', 2, 'box', 'on')
set(gca, 'fontsize', 20)

xlabel("Angle", 'fontsize', 30)
ylabel("Normalized Power (dB)", 'fontsize', 30)
axis([-90, 90, -30, 0])
legends = {"m=2", "m=4", "m=6", "m=8",  "m=12","m=16", "Half power"};
legend(legends, 'location', 'northeast', 'box', 'off', 'numcolumns', 4, 'location', 'northoutside')
pbaspect([6,3,1])
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/array_beam_width.eps')

% figure(2)
% clf
% plot(1:16, hpbw(1:16), 'linewidth', 2)
% saveas(gcf, 'figs/matlab_test.png')