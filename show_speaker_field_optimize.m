figure(1)
clf
counter = 1;
for ti = 11 : 10 : 61

    speaker.weights_out = speaker_w(ti, :);
    field = show_sound_field_optimize(speaker);
    % keyboard
    % pause(0.1)
    subplot(2,3,counter)
    imagesc(abs(flipud(field)))
    title(ti-1+30)
    pbaspect([1,1,1])
    counter = counter + 1;
end

saveas(gcf, 'figs/matlab_test.png')