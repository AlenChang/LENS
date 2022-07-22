figure(1)
clf
load lens_amp_no_comp.mat

% amps = amps / max(abs(amps));
hold on
plot(freq, amps, 'linewidth', 2)

load lens_amp_comp_speaker4.mat
plot(freq, amps, 'linewidth', 2)

load lens_amp_comp_speaker9.mat
plot(freq, amps, 'linewidth', 2)

load lens_amp_comp_speaker4_1lambda.mat
plot(freq, amps, 'linewidth', 2)

legend("Speaker = 1", "Speaker = 4", "Speaker = 9", "Speaker = 4(1\lambda)",...
    'location', 'best', 'box', 'off')
set(gca, 'fontsize', 20)