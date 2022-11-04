
direction = 90;

params.c = 343;
params.fc = 20e3;
params.lambda = params.c / params.fc;
%% define sound field
field_len = 100;
gridsize = params.lambda / 20;
field = build_sound_field(field_len, field_len, gridsize);

%% STEP1 define speakers
num_speakers = 6;
speaker_center = [field_len/2, 0];
speaker_spacing = 0.5; % 0.5 *lambda
fc = 20e3;
field_type = '1D';
speaker = build_speakers(num_speakers, speaker_center, speaker_spacing, fc, field_type);
speaker.weights_out = speaker_weights(direction+1, :);

%% compute sound field
field = compute_field_pressure(field, speaker);

speaker_pressure = abs(field.sound_pressure).^2;

%% define lens
num_lens = 16;
lens_center = [field_len/2, 0];
lens_spacing = 0.5; % 0.5 *lambda
fc = 20e3;
field_type = '2D';
lens = build_speakers(num_lens, lens_center, lens_spacing, fc, field_type);
lens.weights_out = lens_out(direction+1, :);
field = compute_field_pressure(field, lens);

lens_pressure = abs(field.sound_pressure).^2;

cmap = parula;
max_lens_pressure = max(lens_pressure(:));
max_speaker_pressure = max(speaker_pressure(:));

cmap1 = cmap;
cmap2 = cmap(1:round(max_speaker_pressure/max_lens_pressure * end), :);

lens_pressure = lens_pressure / max_lens_pressure;
speaker_pressure = speaker_pressure / max_lens_pressure;

figure(1)
clf
colormap(cmap2)
imagesc(speaker_pressure)
set(gca, 'xticklabel', [], 'yticklabel', [])
pbaspect([1,1,1])
set(gca, 'fontsize', 40)
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', sprintf('fig_results/visu_wo_lens%i.eps', direction))


figure(2)
clf
colormap(cmap1)
imagesc(lens_pressure)
set(gca, 'xticklabel', [], 'yticklabel', [])
pbaspect([1,1,1])
colorbar
set(gca, 'fontsize', 40)
saveas(gcf, 'figs/matlab_test2.png')
print('-depsc', sprintf('fig_results/visu_w_lens%i.eps', direction))


