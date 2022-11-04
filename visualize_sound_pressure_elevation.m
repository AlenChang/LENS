
direction = 90;

%% define sound field
field_len = 100;
gridsize = params.lambda / 20;
field = build_sound_field(field_len, field_len, gridsize);

field.locs(:, :, 3) = field.locs(:, :, 1)+50;
% field.locs(:, :, 1) = field.locs(:, :, 2);
field.locs(:, :, 1) = 0;
% field.locs(:, :, 2) = field.locs(:, :, 2) + 100;

% for ti = 1 : size(field.locs, 1)
%     for mi = 1 : size(field.locs, 2)
%         field.locs(ti, mi, 3) = field.locs(ti, mi, 2);
%         field.locs(ti, mi, 2) = field.locs(ti, mi, 1) * cos(direction / 180 * pi);
%         field.locs(ti, mi, 1) = field.locs(ti, mi, 1) * sin(direction / 180 * pi);
%     end
% end

%% STEP1 define speakers
num_speakers = 6;
speaker_center = [field_len/2, 0];
speaker_spacing = 0.5; % 0.5 *lambda
fc = 20e3;
field_type = '1D';
speaker = build_speakers(num_speakers, speaker_center, speaker_spacing, fc, field_type);

speaker.locs = speaker_locs;

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

lens.locs = lens_locs;

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

figure(2)
clf
lens_phase = reshape(lens.weights_out, 16, []);
imagesc(abs(lens_phase))
saveas(gcf, 'figs/matlab_test.png')
% keyboard

figure(3)
clf
plot3(lens_locs(:, 1), lens_locs(:, 2), lens_locs(:, 3), '*')
hold on
locs = field.locs;
locs = reshape(locs, [], 3);
plot3(locs(:, 1), locs(:, 2), locs(:, 3), '*')
axis([-50, 50, -50, 50, 0, 100])
xlabel("X")
ylabel("Y")
zlabel("Z")
saveas(gcf, 'figs/matlab_position.png')
% keyboard

figure(1)
clf
colormap(cmap2)
imagesc(speaker_pressure)
set(gca, 'xticklabel', [], 'yticklabel', [])
pbaspect([1,1,1])
set(gca, 'fontsize', 40)
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', sprintf('fig_results/ele_visu_wo_lens%i.eps', direction))


figure(2)
clf
colormap(cmap1)
imagesc(lens_pressure)
set(gca, 'xticklabel', [], 'yticklabel', [])
pbaspect([1,1,1])
colorbar
set(gca, 'fontsize', 40)
saveas(gcf, 'figs/matlab_test2.png')
print('-depsc', sprintf('fig_results/ele_visu_w_lens%i.eps', direction))


